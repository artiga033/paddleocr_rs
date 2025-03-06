use image::{DynamicImage, GenericImageView, GrayImage, Luma};
use imageproc::{point::Point, rect::Rect};
use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use ort::inputs;
use ort::session::{builder::SessionBuilder, Session};
use std::path::Path;

use crate::{error::PaddleOcrResult, PaddleOcrError};

pub struct Det {
    model: Session,
    rect_border_size: u32,
}

impl Det {
    const RECT_BORDER_SIZE: u32 = 8;

    pub fn new(model: Session) -> Self {
        Self {
            model,
            rect_border_size: Self::RECT_BORDER_SIZE,
        }
    }

    pub fn from_file(model_path: impl AsRef<Path>) -> PaddleOcrResult<Self> {
        let model = SessionBuilder::new()?
            .with_execution_providers([
                #[cfg(feature = "ort-cuda")]
                ort::execution_providers::cuda::CUDAExecutionProvider::default().build(),
                #[cfg(feature = "ort-tensorrt")]
                ort::execution_providers::tensorrt::TensorRTExecutionProvider::default().build(),
                #[cfg(feature = "ort-openvino")]
                ort::execution_providers::openvino::OpenVINOExecutionProvider::default().build(),
                #[cfg(feature = "ort-onednn")]
                ort::execution_providers::onednn::OneDNNExecutionProvider::default().build(),
                #[cfg(feature = "ort-directml")]
                ort::execution_providers::directml::DirectMLExecutionProvider::default().build(),
                #[cfg(feature = "ort-qnn")]
                ort::execution_providers::qnn::QnnExecutionProvider::default().build(),
                #[cfg(feature = "ort-coreml")]
                ort::execution_providers::coreml::CoreMLExecutionProvider::default().build(),
                #[cfg(feature = "ort-acl")]
                ort::execution_providers::acl::ACLExecutionProvider::default().build(),
                #[cfg(feature = "ort-tvm")]
                ort::execution_providers::tvm::TVMExecutionProvider::default().build(),
                #[cfg(feature = "ort-cann")]
                ort::execution_providers::cann::CANNExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)?;
        Ok(Self {
            model,
            rect_border_size: Self::RECT_BORDER_SIZE,
        })
    }

    pub fn with_rect_border_size(mut self, rect_border_size: u32) -> Self {
        self.rect_border_size = rect_border_size;
        self
    }

    pub fn find_text_rect(&self, img: &DynamicImage) -> PaddleOcrResult<Vec<Rect>> {
        let input = Self::preprocess(img)?;
        let output = self.run_model(&input, img.width(), img.height())?;
        Ok(self.find_box(&output))
    }

    pub fn find_text_img(&self, img: &DynamicImage) -> PaddleOcrResult<Vec<DynamicImage>> {
        Ok(self
            .find_text_rect(img)?
            .iter()
            .map(|r| img.crop_imm(r.left() as u32, r.top() as u32, r.width(), r.height()))
            .collect())
    }

    fn preprocess(
        img: &DynamicImage,
    ) -> PaddleOcrResult<ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>> {
        let (w, h) = img.dimensions();
        let pad_w = Self::get_pad_length(w);
        let pad_h = Self::get_pad_length(h);

        let mut input = Array::zeros((1, 3, pad_h as usize, pad_w as usize));
        for pixel in img.pixels() {
            let x = pixel.0 as _;
            let y = pixel.1 as _;
            let [r, g, b, _] = pixel.2 .0;
            input[[0, 0, y, x]] = (((r as f32) / 255.) - 0.485) / 0.229;
            input[[0, 1, y, x]] = (((g as f32) / 255.) - 0.456) / 0.224;
            input[[0, 2, y, x]] = (((b as f32) / 255.) - 0.406) / 0.225;
        }
        Ok(input)
    }

    fn run_model(
        &self,
        input: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>,
        width: u32,
        height: u32,
    ) -> PaddleOcrResult<GrayImage> {
        let pad_h = Self::get_pad_length(height);
        let outputs = self.model.run(inputs!["x" => input.view()]?)?;
        let output = outputs
            .iter()
            .next()
            .ok_or(PaddleOcrError::custom("no output"))?
            .1;
        let output = output.try_extract_tensor::<f32>()?.view().t().to_owned();
        let output: Vec<_> = output.iter().collect();
        let img = image::ImageBuffer::from_fn(width, height, |x, y| {
            Luma([(*output[(x * pad_h + y) as usize] * 255.0).min(255.0) as u8])
        });
        Ok(img)
    }

    fn find_box(&self, img: &GrayImage) -> Vec<Rect> {
        let (w, h) = img.dimensions();
        imageproc::contours::find_contours_with_threshold::<u32>(img, 200)
            .into_iter()
            .filter(|x| x.parent.is_none())
            .map(|x| x.points)
            .filter_map(|x| Self::bounding_rect(&x))
            .map(|x| {
                Rect::at(
                    (x.left() - self.rect_border_size as i32).max(0),
                    (x.top() - self.rect_border_size as i32).max(0),
                )
                .of_size(
                    (x.width() + self.rect_border_size * 2).min(w),
                    (x.height() + self.rect_border_size * 2).min(h),
                )
            })
            .collect()
    }

    fn bounding_rect(points: &[Point<u32>]) -> Option<Rect> {
        let (x_min, x_max, y_min, y_max) = points.iter().fold(None, |ret, p| match ret {
            None => Some((p.x, p.x, p.y, p.y)),
            Some((x_min, x_max, y_min, y_max)) => Some((
                x_min.min(p.x),
                x_max.max(p.x),
                y_min.min(p.y),
                y_max.max(p.y),
            )),
        })?;
        let width = x_max - x_min;
        let height = y_max - y_min;
        if width <= 5 || height <= 5 {
            return None;
        }
        Some(Rect::at(x_min as i32, y_min as i32).of_size(width, height))
    }

    const fn get_pad_length(length: u32) -> u32 {
        let i = length % 32;
        if i == 0 {
            length
        } else {
            length + 32 - i
        }
    }
}

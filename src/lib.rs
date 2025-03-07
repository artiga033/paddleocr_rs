mod det;
mod error;
mod rec;

pub use det::Det;
pub use error::PaddleOcrError;
pub use rec::Rec;

#[cfg(test)]
mod tests {
    use tracing_subscriber::filter::LevelFilter;

    use super::*;

    #[test]
    fn it_works() -> Result<(), Box<dyn std::error::Error>> {
        tracing_subscriber::fmt()
            .with_max_level(LevelFilter::DEBUG)
            .init();
        let det = Det::from_file("./models/ch_PP-OCRv4_det_infer.onnx")?;
        let rec = Rec::from_file(
            "./models/ch_PP-OCRv4_rec_infer.onnx",
            "./models/ppocr_keys_v1.txt",
        )?;
        let img = image::open("./test/test.png")?;
        for sub in det.find_text_img(&img)? {
            println!("{}", rec.predict_str(&sub)?)
        }
        Ok(())
    }
}

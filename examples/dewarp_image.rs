//! Prevedie obrázok na rovný (dewarp): `dewarp_image <dewarp.json> <vstup.jpg> <výstup.jpg>`
use object_detect::dewarp::{Dewarp, DewarpParams};
use opencv::{core::Vector, imgcodecs, prelude::*};
fn main() -> object_detect::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let p = DewarpParams::load(&a[1])?;
    let img = imgcodecs::imread(&a[2], imgcodecs::IMREAD_COLOR)?;
    let out = Dewarp::new(&p, img.size()?)?.apply(&img)?;
    imgcodecs::imwrite(&a[3], &out, &Vector::new())?;
    Ok(())
}

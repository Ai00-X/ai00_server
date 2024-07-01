use std::cmp::Ordering;
use voracious_radix_sort::Radixable;
#[derive(Copy, Clone, Debug)]
pub struct F32WithIndex(pub usize, pub f32);
impl PartialOrd for F32WithIndex {
    fn partial_cmp(&self, other: &F32WithIndex) -> Option<Ordering> {
        self.1.partial_cmp(&other.1)
    }
}
impl PartialEq for F32WithIndex {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}
impl Radixable<f32> for F32WithIndex {
    type Key = f32;
    #[inline]
    fn key(&self) -> Self::Key {
        self.1
    }
}
#[derive(Copy, Clone, Debug)]
pub struct DoubleF32WithIndex(pub usize, pub f32, pub f32);
impl PartialOrd for DoubleF32WithIndex {
    fn partial_cmp(&self, other: &DoubleF32WithIndex) -> Option<Ordering> {
        self.2.partial_cmp(&other.2)
    }
}
impl PartialEq for DoubleF32WithIndex {
    fn eq(&self, other: &Self) -> bool {
        self.2 == other.2
    }
}
impl Radixable<f32> for DoubleF32WithIndex {
    type Key = f32;
    #[inline]
    fn key(&self) -> Self::Key {
        self.2
    }
}

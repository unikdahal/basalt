//! `NativeType` — the safety contract for raw byte reinterpretation of
//! fixed-width primitives. See design-docs/basalt-phase2-lld.md §3.1/§3.3.

/// Marks a type as safe to reinterpret as, or construct from, a raw byte
/// slice of `size_of::<Self>()` bytes.
///
/// # Safety
/// Implementing this trait asserts that `Self` has no padding bytes, is
/// valid for any bit pattern, and that copying its bytes is equivalent to
/// copying the value (no interior pointers, no niches). Every fixed-width
/// numeric primitive below satisfies this. Do not implement it for types
/// with padding, niche optimizations, or non-`'static` borrowed data.
pub unsafe trait NativeType: Copy + Send + Sync + 'static {}

unsafe impl NativeType for i8 {}
unsafe impl NativeType for i16 {}
unsafe impl NativeType for i32 {}
unsafe impl NativeType for i64 {}
unsafe impl NativeType for u8 {}
unsafe impl NativeType for u16 {}
unsafe impl NativeType for u32 {}
unsafe impl NativeType for u64 {}
unsafe impl NativeType for f32 {}
unsafe impl NativeType for f64 {}

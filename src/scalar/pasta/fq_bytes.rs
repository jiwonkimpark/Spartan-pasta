use std::fmt::Debug;
use std::ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign};
use ff::{Field, FromUniformBytes};
use rand_core::CryptoRngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};
use crate::scalar::pasta::fq::{Bytes, Fq, R};

type UnpackedFqBytes = Fq;

#[derive(Copy, Clone, Hash)]
pub struct FqBytes {
    /// `bytes` is a little-endian byte encoding of an integer representing a scalar modulo the
    /// group order.
    pub(crate) bytes: [u8; 32],
}

impl FqBytes {
    /// Construct a `FqBytes` by reducing a 256-bit little-endian integer
    /// modulo the group order \\( \q \\).
    pub fn from_bytes_mod_order(bytes: [u8; 32]) -> Self {
        // Temporarily allow s_unreduced.bytes > 2^255 ...
        let unreduced = FqBytes { bytes };

        // Then reduce mod the group order and return the reduced representative.
        let s = unreduced.reduce();
        // debug_assert_eq!(0u8, s[31] >> 7);

        s
    }

    /// Construct a `FqBytes` by reducing a 512-bit little-endian integer
    /// modulo the group order \\( \q \\).
    pub fn from_bytes_mod_order_wide(input: &[u8; 64]) -> FqBytes {
        UnpackedFqBytes::from_uniform_bytes(input).pack()
    }

    /// Attempt to construct a `FqBytes` from a canonical byte representation.
    ///
    /// # Return
    ///
    /// - `Some(s)`, where `s` is the `FqBytes` corresponding to `bytes`,
    ///   if `bytes` is a canonical byte representation modulo the group order \\( \q \\);
    /// - `None` if `bytes` is not a canonical byte representation.
    pub fn from_canonical_bytes(bytes: [u8; 32]) -> CtOption<FqBytes> {
        let high_bit_unset = (bytes[31] >> 7).ct_eq(&0); // TODO: check if we need this
        let candidate = FqBytes { bytes };
        CtOption::new(candidate, high_bit_unset & candidate.is_canonical())
    }
}

impl Debug for FqBytes {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        write!(f, "FqBytes{{\n\tbytes: {:?},\n}}", &self.bytes)
    }
}

impl Eq for FqBytes {}

impl PartialEq for FqBytes {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).into()
    }
}

impl ConstantTimeEq for FqBytes {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.bytes.ct_eq(&other.bytes)
    }
}

impl Index<usize> for FqBytes {
    type Output = u8;

    /// Index the bytes of the representative for this `FqBytes`.  Mutation is not permitted.
    fn index(&self, _index: usize) -> &u8 {
        &(self.bytes[_index])
    }
}

impl<'b> MulAssign<&'b FqBytes> for FqBytes {
    fn mul_assign(&mut self, _rhs: &'b FqBytes) {
        *self = UnpackedFqBytes::mul(&self.unpack(), &_rhs.unpack()).pack();
    }
}

/// Define non-borrow variants of `MulAssign`.
macro_rules! define_mul_assign_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty) => {
        impl MulAssign<$rhs> for $lhs {
            fn mul_assign(&mut self, rhs: $rhs) {
                *self *= &rhs;
            }
        }
    };
}

define_mul_assign_variants!(LHS = FqBytes, RHS = FqBytes);

impl<'a, 'b> Mul<&'b FqBytes> for &'a FqBytes {
    type Output = FqBytes;
    fn mul(self, _rhs: &'b FqBytes) -> FqBytes {
        UnpackedFqBytes::mul(&self.unpack(), &_rhs.unpack()).pack()
    }
}

macro_rules! define_mul_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty, Output = $out:ty) => {
        impl<'b> Mul<&'b $rhs> for $lhs {
            type Output = $out;
            fn mul(self, rhs: &'b $rhs) -> $out {
                &self * rhs
            }
        }

        impl<'a> Mul<$rhs> for &'a $lhs {
            type Output = $out;
            fn mul(self, rhs: $rhs) -> $out {
                self * &rhs
            }
        }

        impl Mul<$rhs> for $lhs {
            type Output = $out;
            fn mul(self, rhs: $rhs) -> $out {
                &self * &rhs
            }
        }
    };
}

define_mul_variants!(LHS = FqBytes, RHS = FqBytes, Output = FqBytes);

impl<'b> AddAssign<&'b FqBytes> for FqBytes {
    fn add_assign(&mut self, _rhs: &'b FqBytes) {
        *self = *self + _rhs;
    }
}

macro_rules! define_add_assign_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty) => {
        impl AddAssign<$rhs> for $lhs {
            fn add_assign(&mut self, rhs: $rhs) {
                *self += &rhs;
            }
        }
    };
}

define_add_assign_variants!(LHS = FqBytes, RHS = FqBytes);

impl<'a, 'b> Add<&'b FqBytes> for &'a FqBytes {
    type Output = FqBytes;
    #[allow(non_snake_case)]
    fn add(self, _rhs: &'b FqBytes) -> FqBytes {
        // The UnpackedScalar::add function produces reduced outputs if the inputs are reduced. By
        // Scalar invariant #1, this is always the case.
        UnpackedFqBytes::add(&self.unpack(), &_rhs.unpack()).pack()
    }
}


/// Define borrow and non-borrow variants of `Add`.
macro_rules! define_add_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty, Output = $out:ty) => {
        impl<'b> Add<&'b $rhs> for $lhs {
            type Output = $out;
            fn add(self, rhs: &'b $rhs) -> $out {
                &self + rhs
            }
        }

        impl<'a> Add<$rhs> for &'a $lhs {
            type Output = $out;
            fn add(self, rhs: $rhs) -> $out {
                self + &rhs
            }
        }

        impl Add<$rhs> for $lhs {
            type Output = $out;
            fn add(self, rhs: $rhs) -> $out {
                &self + &rhs
            }
        }
    };
}

define_add_variants!(LHS = FqBytes, RHS = FqBytes, Output = FqBytes);

impl<'b> SubAssign<&'b FqBytes> for FqBytes {
    fn sub_assign(&mut self, _rhs: &'b FqBytes) {
        *self = *self - _rhs;
    }
}

/// Define non-borrow variants of `SubAssign`.
macro_rules! define_sub_assign_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty) => {
        impl SubAssign<$rhs> for $lhs {
            fn sub_assign(&mut self, rhs: $rhs) {
                *self -= &rhs;
            }
        }
    };
}
define_sub_assign_variants!(LHS = FqBytes, RHS = FqBytes);

impl<'a, 'b> Sub<&'b FqBytes> for &'a FqBytes {
    type Output = FqBytes;
    #[allow(non_snake_case)]
    fn sub(self, rhs: &'b FqBytes) -> FqBytes {
        // The UnpackedScalar::sub function produces reduced outputs if the inputs are reduced. By
        // Scalar invariant #1, this is always the case.
        UnpackedFqBytes::sub(&self.unpack(), &rhs.unpack()).pack()
    }
}

/// Define borrow and non-borrow variants of `Sub`.
macro_rules! define_sub_variants {
    (LHS = $lhs:ty, RHS = $rhs:ty, Output = $out:ty) => {
        impl<'b> Sub<&'b $rhs> for $lhs {
            type Output = $out;
            fn sub(self, rhs: &'b $rhs) -> $out {
                &self - rhs
            }
        }

        impl<'a> Sub<$rhs> for &'a $lhs {
            type Output = $out;
            fn sub(self, rhs: $rhs) -> $out {
                self - &rhs
            }
        }

        impl Sub<$rhs> for $lhs {
            type Output = $out;
            fn sub(self, rhs: $rhs) -> $out {
                &self - &rhs
            }
        }
    };
}

define_sub_variants!(LHS = FqBytes, RHS = FqBytes, Output = FqBytes);

impl<'a> Neg for &'a FqBytes {
    type Output = FqBytes;
    #[allow(non_snake_case)]
    fn neg(self) -> FqBytes {
        // TODO: mul_internal is not implemented in Fq
        let reduced = UnpackedFqBytes::mul(&self.unpack(), &R);
        // let self_R = UnpackedFqBytes::mul_internal(&self.unpack(), &constants::R);
        // let self_mod_l = UnpackedFqBytes::montgomery_reduce(&self_R);
        UnpackedFqBytes::sub(&UnpackedFqBytes::ZERO, &reduced).pack()
    }
}

impl Neg for FqBytes {
    type Output = FqBytes;
    fn neg(self) -> FqBytes {
        -&self
    }
}

impl ConditionallySelectable for FqBytes {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let mut bytes = [0u8; 32];
        #[allow(clippy::needless_range_loop)]
        for i in 0..32 {
            bytes[i] = u8::conditional_select(&a.bytes[i], &b.bytes[i], choice);
        }
        FqBytes { bytes }
    }
}

// #[cfg(feature = "serde")]
use serde::de::Visitor;
// #[cfg(feature = "serde")]
use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

// #[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl Serialize for FqBytes {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
    {
        use serde::ser::SerializeTuple;
        let mut tup = serializer.serialize_tuple(32)?;
        for byte in self.as_bytes().iter() {
            tup.serialize_element(byte)?;
        }
        tup.end()
    }
}

//#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'de> Deserialize<'de> for FqBytes {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
    {
        struct FqBytesVisitor;

        impl<'de> Visitor<'de> for FqBytesVisitor {
            type Value = FqBytes;

            fn expecting(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                formatter.write_str(
                    "a sequence of 32 bytes whose little-endian interpretation is less than the \
                    base point order q",
                )
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<FqBytes, A::Error>
                where
                    A: serde::de::SeqAccess<'de>,
            {
                let mut bytes = [0u8; 32];
                #[allow(clippy::needless_range_loop)]
                for i in 0..32 {
                    bytes[i] = seq
                        .next_element()?
                        .ok_or_else(|| serde::de::Error::invalid_length(i, &"expected 32 bytes"))?;
                }
                Option::from(FqBytes::from_canonical_bytes(bytes))
                    .ok_or_else(|| serde::de::Error::custom("scalar was not canonically encoded"))
            }
        }

        deserializer.deserialize_tuple(32, FqBytesVisitor)
    }
}

impl FqBytes {
    pub const ZERO: Self = Self { bytes: [0u8; 32] };

    pub const ONE: Self = Self {
        bytes: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    };

    #[cfg(any(test, feature = "rand_core"))]
    /// Return a `Scalar` chosen uniformly at random using a user-provided RNG.
    ///
    /// # Inputs
    ///
    /// * `rng`: any RNG which implements `CryptoRngCore`
    ///   (i.e. `CryptoRng` + `RngCore`) interface.
    pub fn random<R: CryptoRngCore + ?Sized>(rng: &mut R) -> Self {
        let mut bytes = [0u8; 64];
        rng.fill_bytes(&mut bytes);
        FqBytes::from_bytes_mod_order_wide(&bytes)
    }

    /// Convert this `FqBytes` to its underlying sequence of bytes.
    pub const fn to_bytes(&self) -> [u8; 32] {
        self.bytes
    }


    /// View the little-endian byte encoding of the integer representing this FqBytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }

    /// Given a nonzero `FqBytes`, compute its multiplicative inverse.
    ///
    /// # Warning
    ///
    /// `self` **MUST** be nonzero.  If you cannot
    /// *prove* that this is the case, you **SHOULD NOT USE THIS
    /// FUNCTION**.
    ///
    /// # Returns
    ///
    /// The multiplicative inverse of this `FqBytes`.
    pub fn invert(&self) -> FqBytes {
        self.unpack().invert().unwrap().pack()
    }

    /// Unpack this `FqBytes` to an `UnpackedFqBytes` for faster arithmetic.
    pub(crate) fn unpack(&self) -> UnpackedFqBytes {
        UnpackedFqBytes::from_bytes(&self.bytes).unwrap()
    }


    /// Reduce this `FqBytes` modulo \\(\q\\).
    #[allow(non_snake_case)]
    fn reduce(&self) -> FqBytes {
        // TODO: need to check if
        let x = self.unpack();
        let reduced = UnpackedFqBytes::mul(&x, &R); // since Fq::mul multiplies and converts to montgomery form.
        // let xR = UnpackedFqBytes::mul_internal(&x, &constants::R);
        // let x_mod_l = UnpackedFqBytes::montgomery_reduce(&xR); // xRR^{-1} mod l = x mod l
        reduced.pack()
    }


    /// Check whether this `FqBytes` is the canonical representative mod \\(\q\\). This is not
    /// public because any `FqBytes` that is publicly observed is reduced, by scalar invariant #2.
    fn is_canonical(&self) -> Choice {
        self.ct_eq(&self.reduce())
    }
}

impl UnpackedFqBytes {
    /// Pack the limbs of this `UnpackedFqBytes` into a `FqBytes`.
    fn pack(&self) -> FqBytes {
        FqBytes {
            bytes: self.as_bytes(),
        }
    }
}

// TODO: test code
#[cfg(test)]
pub(crate) mod test {
    use crate::scalar::pasta::fq_bytes::FqBytes;

    pub(crate) static LARGEST_UNREDUCED_SCALAR: FqBytes = FqBytes {
        bytes: [
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0x7f,
        ],
    };

    const BASEPOINT_ORDER: FqBytes = FqBytes {
        bytes: [
            0x00, 0x00, 0x00, 0x00, 0x21, 0xeb, 0x46, 0x8c, 0xdd, 0xa8, 0x94, 0x09, 0xfc, 0x98,
            0x46, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x40,
        ],
    };

    #[test]
    fn test_to_bytes() {

    }

    #[test]
    fn test_reduce() {
        let reduced = FqBytes::from_bytes_mod_order(BASEPOINT_ORDER.bytes);
        println!("{:?}", reduced.bytes)
    }
}
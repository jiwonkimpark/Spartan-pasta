use super::errors::ProofVerifyError;
use super::scalar::{Scalar};
use core::ops::{Mul, MulAssign};
use std::borrow::Borrow;
use std::ops::{Add, Sub};
use ff::PrimeField;
use pasta_curves::{Ep};
use pasta_curves::arithmetic::CurveExt;
use pasta_curves::group::Group;
use pasta_curves::pallas::Affine;
use subtle::{Choice, ConditionallySelectable};
use crate::compression::{CompressedPoint, PALLAS_GENERATOR_COMPRESSED};
use crate::scalar::pasta::fq::Fq;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GroupElement(pub pasta_curves::pallas::Point);
pub type Point = pasta_curves::pallas::Point;
pub type CompressedGroup = CompressedPoint; // curve25519_dalek::ristretto::CompressedRistretto;

impl GroupElement {
    pub fn generator() -> Self {
        GroupElement(Point::generator())
    }

    pub fn into(&self) -> Point {
        self.0
    }

    pub fn from_affine(affine: Affine) -> Self {
        GroupElement(Point::from(affine))
    }

    pub fn compress(&self) -> CompressedGroup {
        CompressedPoint::compress(self.0)
    }

    pub fn from_uniform_bytes(bytes: &[u8; 64]) -> Self {
        let hash = Point::hash_to_curve("commitment");
        let point = hash(bytes);

        GroupElement(point)
    }
}

// impl Borrow<GroupElement> for Ep {
//     fn borrow(&self) -> &GroupElement {
//         &GroupElement::from(self)
//     }
// }

pub trait CompressedGroupExt {
    type Group;
    fn unpack(&self) -> Result<Self::Group, ProofVerifyError>;
}

impl CompressedGroupExt for CompressedGroup {
    type Group = Ep;
    fn unpack(&self) -> Result<Self::Group, ProofVerifyError> {
        self
            .decompress()
            .ok_or_else(|| ProofVerifyError::DecompressionError(self.to_bytes()))
    }
}

pub const GROUP_BASEPOINT_COMPRESSED: CompressedGroup =
    PALLAS_GENERATOR_COMPRESSED;

// The following code was copied from pasta_curves:
// https://github.com/zcash/pasta_curves/blob/main/src/curves.rs
impl<'a, 'b> Mul<&'b Fq> for &'a Ep {
    type Output = Ep;

    fn mul(self, rhs: &'b Fq) -> Self::Output {
        let mut acc = Self::Output::identity();

        // This is a simple double-and-add implementation of point
        // multiplication, moving from most significant to least
        // significant bit of the scalar.
        //
        // We don't use `PrimeFieldBits::.to_le_bits` here, because that would
        // force users of this crate to depend on `bitvec` where they otherwise
        // might not need to.
        //
        // NOTE: We skip the leading bit because it's always unset (we are turning
        // the 32-byte repr into 256 bits, and $scalar::NUM_BITS = 255).

        for bit in rhs
            .to_repr().iter().rev().flat_map(|byte| (0..8).rev().map(move |i| Choice::from((byte >> i) & 1u8)))
            .skip(1) {
            acc = acc.double();
            acc = Self::Output::conditional_select(&acc, &(acc + self), bit)
        }
        acc
    }
}

impl<'a, 'b> Mul<&'b Ep> for &'a Fq {
    type Output = Ep;

    fn mul(self, rhs: &'b Ep) -> Self::Output {
        // This is a simple double-and-add implementation of point
        // multiplication, moving from most significant to least
        // significant bit of the scalar.
        //
        // We don't use `PrimeFieldBits::.to_le_bits` here, because that would
        // force users of this crate to depend on `bitvec` where they otherwise
        // might not need to.
        //
        // NOTE: We skip the leading bit because it's always unset (we are turning
        // the 32-byte repr into 256 bits, and $scalar::NUM_BITS = 255).

        rhs.mul(self)
    }
}

/// For the following three operations, decompress scalar and directly applied scalar
/// TODO: need to check if we need decompressed scalar
impl<'b> MulAssign<&'b Fq> for GroupElement {
    fn mul_assign(&mut self, scalar: &'b Fq) {
        let result = &(self as &GroupElement).0 * scalar;
        *self = GroupElement(result);
    }
}

impl<'a, 'b> Mul<&'b Scalar> for &'a GroupElement {
    type Output = GroupElement;
    fn mul(self, scalar: &'b Scalar) -> GroupElement {
        let result = &self.0 * scalar;
        GroupElement(result)
    }
}

impl<'a, 'b> Mul<&'b GroupElement> for &'a Scalar {
    type Output = GroupElement;

    fn mul(self, point: &'b GroupElement) -> GroupElement {
        let result = &point.0 * self;
        GroupElement(result)
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

macro_rules! define_mul_assign_variants {
  (LHS = $lhs:ty, RHS = $rhs:ty) => {
    impl MulAssign<$rhs> for $lhs {
      fn mul_assign(&mut self, rhs: $rhs) {
        *self *= &rhs;
      }
    }
  };
}

define_mul_assign_variants!(LHS = GroupElement, RHS = Scalar);
define_mul_variants!(LHS = Fq, RHS = Ep, Output = Ep);
define_mul_variants!(LHS = GroupElement, RHS = Scalar, Output = GroupElement);
define_mul_variants!(LHS = Scalar, RHS = GroupElement, Output = GroupElement);

impl<'a, 'b> Add<&'b GroupElement> for &'a GroupElement {
    type Output = GroupElement;

    fn add(self, rhs: &'b GroupElement) -> Self::Output {
        GroupElement(&self.0 + rhs.0)
    }
}
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

impl<'a, 'b> Sub<&'b GroupElement> for &'a GroupElement {
    type Output = GroupElement;

    fn sub(self, other: &'b GroupElement) -> Self::Output {
        GroupElement(&self.0 - &other.0)
    }
}

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

define_add_variants!(LHS = GroupElement, RHS = GroupElement, Output = GroupElement);
define_sub_variants!(LHS = GroupElement, RHS = GroupElement, Output = GroupElement);

pub trait VartimeMultiscalarMul {
    type Scalar;
    fn vartime_multiscalar_mul<I, J>(scalars: I, points: J) -> Self
        where
            I: IntoIterator,
            I::Item: Borrow<Self::Scalar>,
            J: IntoIterator,
            J::Item: Borrow<Self>,
            Self: Clone;
}

impl VartimeMultiscalarMul for GroupElement {
    type Scalar = Scalar;

    // TODO: make borrow I and J
    // where
    // I::Item: Borrow<Self::Scalar>,
    // J::Item: Borrow<Self>,
    fn vartime_multiscalar_mul<I, J>(scalars: I, points: J) -> Self
        where
            I: IntoIterator,
            I::Item: Borrow<Self::Scalar>,
            J: IntoIterator,
            J::Item: Borrow<Self>,
            Self: Clone,
    {
        // TODO: use chain
        let mut point_iter = points.into_iter();
        let mut scalars_iter = scalars.into_iter();

        let (p_lo, p_hi) = point_iter.by_ref().size_hint();
        let (s_lo, s_hi) = scalars_iter.by_ref().size_hint();

        assert_eq!(p_lo, s_lo);
        assert_eq!(p_hi, Some(p_lo));
        assert_eq!(s_hi, Some(s_lo));

        let points = point_iter.map(|p| p.borrow().into()).collect::<Vec<Point>>();
        let scalars = scalars_iter.map(|s| Fq(s.borrow().0)).collect::<Vec<Scalar>>();
        //
        let len = p_lo;
        //
        let mut sum = GroupElement(Ep::identity());
        for i in 0..len {
            let mul = GroupElement(points[i]) * scalars[i];
            sum = sum + mul;
        }
        //
        // for (point, scalar) in points.into_iter().zip(scalars.into_iter()) {
        //     let mul = point * scalar;
        //     sum = sum + mul;
        // }

        sum
    }
}

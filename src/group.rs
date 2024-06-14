use super::errors::ProofVerifyError;
use super::scalar::{Scalar};
use core::borrow::Borrow;
use core::ops::{Mul, MulAssign};
use std::ops::{Add, Sub};
use pasta_curves::{Ep};
use pasta_curves::group::Group;
use pasta_curves::pallas::Affine;
use crate::compression::{CompressedPoint, PALLAS_GENERATOR_COMPRESSED};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GroupElement(Ep);
pub type Point = Ep;
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
}

pub trait CompressedGroupExt {
    type Group;
    fn unpack(&self) -> Result<Self::Group, ProofVerifyError>;
}

impl CompressedGroupExt for CompressedGroup {
    type Group = pasta_curves::Ep;
    fn unpack(&self) -> Result<Self::Group, ProofVerifyError> {
        self
            .decompress()
            .ok_or_else(|| ProofVerifyError::DecompressionError(self.to_bytes()))
    }
}

pub const GROUP_BASEPOINT_COMPRESSED: CompressedGroup =
    PALLAS_GENERATOR_COMPRESSED;

/// For the following three operations, decompress scalar and directly applied scalar
/// TODO: need to check if we need decompressed scalar
impl<'b> MulAssign<&'b Scalar> for GroupElement {
    fn mul_assign(&mut self, scalar: &'b Scalar) {
        let result = (self as &GroupElement).0.mul(scalar);
        *self = result;
    }
}

impl<'a, 'b> Mul<&'b Scalar> for &'a GroupElement {
    type Output = GroupElement;
    fn mul(self, scalar: &'b Scalar) -> GroupElement {
        self.0.mul(scalar)
    }
}

impl<'a, 'b> Mul<&'b GroupElement> for &'a Scalar {
    type Output = GroupElement;

    fn mul(self, point: &'b GroupElement) -> GroupElement {
        point.0.mul(self)
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
    fn vartime_multiscalar_mul<I, J>(scalars: I, points: J) -> Self
        where
            I: IntoIterator,
            I::Item: Borrow<Self::Scalar>,
            J: IntoIterator,
            J::Item: Borrow<Self>,
            Self: Clone,
    {
        let group_default = GroupElement(Ep::default());

        points.into_iter()
            .zip(scalars.into_iter())
            .fold(group_default, | acc, (point, scalar)| acc + point * scalar)

        // <Self as VartimeMultiscalarMul>::vartime_multiscalar_mul(
        //     scalars
        //         .into_iter()
        //         .map(|s| Scalar::decompress_scalar(s.borrow()))
        //         .collect::<Vec<ScalarBytes>>(),
        //     points,
        // )
    }
}

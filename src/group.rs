use super::errors::ProofVerifyError;
use super::scalar::Scalar;
use core::ops::{Mul, MulAssign};
use std::borrow::Borrow;
use std::ops::{Add, Sub};
use ff::PrimeField;
use group::Curve;
use pasta_curves::{Ep};
use pasta_curves::arithmetic::{CurveAffine, CurveExt};
use pasta_curves::group::Group;
use pasta_curves::pallas::Affine;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
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

impl Serialize for GroupElement {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let compressed = CompressedPoint::compress(self.0);
        compressed.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for GroupElement {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let compressed = CompressedPoint::deserialize(deserializer).expect("failed to deserialize");
        let point = compressed.decompress().unwrap();
        Ok(GroupElement(point))
    }
}

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
define_mul_variants!(LHS = Ep, RHS = Fq, Output = Ep);

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
        let mut point_iter = points.into_iter();
        let mut scalars_iter = scalars.into_iter();

        let (p_lo, p_hi) = point_iter.by_ref().size_hint();
        let (s_lo, s_hi) = scalars_iter.by_ref().size_hint();

        assert_eq!(p_lo, s_lo);
        assert_eq!(p_hi, Some(p_lo));
        assert_eq!(s_hi, Some(s_lo));
        let len = p_lo;

        let points: Vec<Affine> = point_iter.map(|p| p.borrow().into().to_affine()).collect::<Vec<Affine>>();
        let scalars = scalars_iter.map(|s| pasta_curves::Fq::from_repr(Fq(s.borrow().0).to_repr()).unwrap()).collect::<Vec<pasta_curves::Fq>>();

        let result = pasta_msm::pallas(points.as_slice(), scalars.as_slice());

        // let result: Ep = if len >= 128 {
        //     pasta_msm::pallas(points.as_slice(), scalars.as_slice())
        // } else {
        //     msm_best(scalars.as_slice(), points.as_slice())
        // };

        // let result = msm_best(scalars.as_slice(), points.as_slice());
        // let mut pairs = Vec::with_capacity(len);
        // for i in 0..len {
        //     let scalar = pasta_curves::Fq::from_repr(scalars[i].to_repr()).unwrap();
        //     pairs.push((scalar, points[i]));
        // }
        // let sum = multiexp::multiexp_vartime(&pairs);

        GroupElement(result)
    }
}

#[cfg(test)]
mod tests {
    use ff::PrimeField;
    use pasta_curves::{Ep, Fq};
    use pasta_curves::group::Group;
    use crate::group::{GroupElement, VartimeMultiscalarMul};
    use crate::scalar::Scalar;
    use multiexp::multiexp_vartime;

    #[test]
    fn test_scalar_multiplication() {
        let generator = Ep::generator();
        let generator_group_element = GroupElement(generator);

        let one_times_generator = Scalar::one() * generator_group_element;
        assert_eq!(generator, one_times_generator.0)
    }

    #[test]
    fn test_vartime_multiscalar_mul_with_multiexp() {
        let generator = Ep::generator();
        let identity = Ep::identity();
        let mut points: Vec<GroupElement> = Vec::new();
        points.push(GroupElement(generator));
        points.push(GroupElement(identity));
        let mut points_converted: Vec<Ep> = Vec::new();
        for point in points {
            points_converted.push(point.0)
        }

        // test 1: 1 * Generator + 0 * Identity should be Generator
        let one = Scalar::one();
        let zero = Scalar::zero();
        let mut scalars1: Vec<Scalar> = Vec::new();
        scalars1.push(one);
        scalars1.push(zero);

        let mut scalars_converted: Vec<pasta_curves::Fq> = Vec::new();
        for scalar1 in scalars1 {
            let scalar = Fq::from_repr(scalar1.to_repr()).unwrap();
            scalars_converted.push(Fq::from_repr(scalar1.to_repr()).unwrap())
        }

        let len = 1;
        let mut pairs = Vec::with_capacity(len);
        for i in 0..1 {
            pairs.push((scalars_converted[i], points_converted[i]));
        }
        let result = multiexp_vartime(&pairs);
        assert_eq!(result, generator);
    }

    #[test]
    fn test_vartime_multiscalar_mul() {
        let generator = Ep::generator();
        let identity = Ep::identity();
        let mut points: Vec<GroupElement> = Vec::new();
        points.push(GroupElement(generator));
        points.push(GroupElement(identity));

        // test 1: 1 * Generator + 0 * Identity should be Generator
        let one = Scalar::one();
        let zero = Scalar::zero();
        let mut scalars1: Vec<Scalar> = Vec::new();
        scalars1.push(one);
        scalars1.push(zero);

        let result = GroupElement::vartime_multiscalar_mul(scalars1, points.clone());
        assert_eq!(result, GroupElement(generator));

        // test 2: 2 * Generator + 2 * Identity should be 2 * Generator
        let two = one + one;
        let mut scalars2: Vec<Scalar> = Vec::new();
        scalars2.push(two);
        scalars2.push(two);

        let expected = two * generator;
        let result = GroupElement::vartime_multiscalar_mul(scalars2, points.clone());
        assert_eq!(result, GroupElement(expected));

        // test 3: 0 * Generator + 2 * Identity should be Identity
        let mut scalars3: Vec<Scalar> = Vec::new();
        scalars3.push(zero);
        scalars3.push(two);

        let result = GroupElement::vartime_multiscalar_mul(scalars3, points.clone());
        assert_eq!(result, GroupElement(identity));
    }
}

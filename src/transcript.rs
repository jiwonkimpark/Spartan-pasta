use digest::{Digest, Update};
use ff::{FromUniformBytes};
use itertools::Itertools;
use super::group::CompressedGroup;
use super::scalar::Scalar;
use merlin::Transcript;
use sha3::Keccak256;
use crate::scalar::pasta::fq::Bytes;

pub trait ProofTranscript {
  fn append_protocol_name(&mut self, protocol_name: &'static [u8]);
  fn append_scalar(&mut self, label: &'static [u8], scalar: &Scalar);
  fn append_point(&mut self, label: &'static [u8], point: &CompressedGroup);
  fn challenge_scalar(&mut self, label: &'static [u8]) -> Scalar;
  fn challenge_vector(&mut self, label: &'static [u8], len: usize) -> Vec<Scalar>;
}

pub struct Keccak256Transcript {
  round: u16,
  state: [u8; 64],
  transcript: Keccak256,
}

const PERSONA_TAG: &[u8] = b"NoTR";
const DOM_SEP_TAG: &[u8] = b"NoDS";
const KECCAK256_STATE_SIZE: usize = 64;
const KECCAK256_PREFIX_CHALLENGE_LO: u8 = 0;
const KECCAK256_PREFIX_CHALLENGE_HI: u8 = 1;

fn compute_updated_state(keccak_instance: Keccak256, input: &[u8]) -> [u8; KECCAK256_STATE_SIZE] {
  let mut updated_instance = keccak_instance;
  digest::Update::update(&mut updated_instance, input);

  let input_lo = &[KECCAK256_PREFIX_CHALLENGE_LO];
  let input_hi = &[KECCAK256_PREFIX_CHALLENGE_HI];

  let mut hasher_lo = updated_instance.clone();
  let mut hasher_hi = updated_instance;

  digest::Update::update(&mut hasher_lo, input_lo);
  digest::Update::update(&mut hasher_hi, input_hi);

  let output_lo = hasher_lo.finalize();
  let output_hi = hasher_hi.finalize();

  [output_lo, output_hi]
      .concat()
      .as_slice()
      .try_into()
      .unwrap()
}

impl Keccak256Transcript {
  pub fn new(label: &'static [u8]) -> Self {
    let keccak_instance = Keccak256::new();
    let input = [PERSONA_TAG, label].concat();
    let output = compute_updated_state(keccak_instance.clone(), &input);

    Self {
      round: 0u16,
      state: output,
      transcript: keccak_instance,
    }
  }

  pub fn append_message(&mut self, label: &'static [u8], message: &[u8]) {
    digest::Update::update(&mut self.transcript, label);
    digest::Update::update(&mut self.transcript, message);
  }

  pub fn append_u64(&mut self, label: &'static [u8], x: u64) {
    digest::Update::update(&mut self.transcript, label);
    digest::Update::update(&mut self.transcript, &encode_u64(x));
  }
}

fn encode_u64(x: u64) -> [u8; 8] {
  use byteorder::{ByteOrder, LittleEndian};

  let mut buf = [0; 8];
  LittleEndian::write_u64(&mut buf, x);
  buf
}

impl ProofTranscript for Keccak256Transcript {
  fn append_protocol_name(&mut self, protocol_name: &'static [u8]) {
    self.append_message(b"protocol-name", protocol_name);
  }

  fn append_scalar(&mut self, label: &'static [u8], scalar: &Scalar) {
    self.append_message(label, &scalar.to_transcript_bytes());
  }

  fn append_point(&mut self, label: &'static [u8], point: &CompressedGroup) {
    self.append_message(label, point.as_bytes());
  }

  fn challenge_scalar(&mut self, label: &'static [u8]) -> Scalar {
    let input = [
      DOM_SEP_TAG,
      self.round.to_le_bytes().as_ref(),
      self.state.as_ref(),
      label,
    ]
        .concat();
    let output = compute_updated_state(self.transcript.clone(), &input);

    // update state
    self.round = {
      if let Some(v) = self.round.checked_add(1) {
        v
      } else {
        panic!("Transcript error")
      }
    };
    self.state.copy_from_slice(&output);
    self.transcript = Keccak256::new();

    Scalar::from_uniform_bytes(&output)
  }

  fn challenge_vector(&mut self, label: &'static [u8], len: usize) -> Vec<Scalar> {
    (0..len).map(|_i| self.challenge_scalar(label)).collect()
  }
}

impl ProofTranscript for Transcript {
  fn append_protocol_name(&mut self, protocol_name: &'static [u8]) {
    self.append_message(b"protocol-name", protocol_name);
  }

  fn append_scalar(&mut self, label: &'static [u8], scalar: &Scalar) {
    self.append_message(label, &scalar.to_bytes());
  }

  fn append_point(&mut self, label: &'static [u8], point: &CompressedGroup) {
    self.append_message(label, point.as_bytes());
  }

  fn challenge_scalar(&mut self, label: &'static [u8]) -> Scalar {
    let mut buf = [0u8; 64];
    self.challenge_bytes(label, &mut buf);
    Scalar::from_bytes_wide(&buf)
  }

  fn challenge_vector(&mut self, label: &'static [u8], len: usize) -> Vec<Scalar> {
    (0..len)
      .map(|_i| self.challenge_scalar(label))
      .collect::<Vec<Scalar>>()
  }
}

pub trait AppendToTranscript {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript);
  fn append_to_keccak_transcript(&self, label: &'static [u8], transcript: &mut Keccak256Transcript);
}

impl AppendToTranscript for Scalar {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript) {
    transcript.append_scalar(label, self);
  }

  fn append_to_keccak_transcript(&self, label: &'static [u8], transcript: &mut Keccak256Transcript) {
    transcript.append_scalar(label, self);
  }
}

impl AppendToTranscript for [Scalar] {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript) {
    transcript.append_message(label, b"begin_append_vector");
    for item in self {
      transcript.append_scalar(label, item);
    }
    transcript.append_message(label, b"end_append_vector");
  }

  fn append_to_keccak_transcript(&self, label: &'static [u8], transcript: &mut Keccak256Transcript) {
    transcript.append_message(label, b"begin_append_vector");
    for item in self {
      transcript.append_scalar(label, item);
    }
    transcript.append_message(label, b"end_append_vector");
  }
}

impl AppendToTranscript for CompressedGroup {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript) {
    transcript.append_point(label, self);
  }

  fn append_to_keccak_transcript(&self, label: &'static [u8], transcript: &mut Keccak256Transcript) {
    transcript.append_point(label, self);
  }
}

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][] and this project adheres
to [Semantic Versioning][].

[Keep a Changelog]: http://keepachangelog.com/en/1.0.0/
[Semantic Versioning]: http://semver.org/spec/v2.0.0.html

## [Unreleased]

### Fixed

## [0.5.0] - 2021-05-05

- Update minimum required Rust version to Rust 1.51. (#39)

## [0.4.3] - 2018-08-26

- Coalesce back-to-back instructions for small performance
  improvements. (#35)

## [0.4.3] - 2018-08-26

### Fixed

- Coalesce back-to-back instructions for small performance
  improvements. (#24)

### Changes

 - Library is now dual-licensed as MIT or APACHE-2. (#12)

## [0.4.2] - 2018-07-11

### Fixed

- No longer reports false positives in certain cases when input data
  is not aligned on a 16-byte boundary. (#19)

## [0.4.1] - 2018-07-07

### Added

- `Substring` support restored. (#20)

## [0.4.0] - 2018-05-13

### Changed

- Complete rewrite from inline assembly to stable compiler intrinsics.

### Added

- Directly searching through a slice of bytes is now supported.

### Removed

- `Substring` was removed during the rewrite.

# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3]

### Added

- aFRR.py: Awarded capacity datetime series can now be passed as an optional argument of calc_acceptance_tolerance_band 
- If provided, awarded capacity data is used to check the fifth reversal point criterion
- pytest_test.py was added to verify that calculation results are in line with TSO examples

### Changed

- aFRR.py: Excluded reversal point from product change period (in accordance with TSO specifications)

### Fixed

- data.py: Fixed relative import
- README.md: Fixed example code in usage section


## [0.0.2]

### Changed

- Add installation instruction for installation the package from pypi

## [0.0.1]

### Added

- Release-based publishing workflow [#1](https://github.com/energy2market/afrr-remuneration/issues/1)
- Initial code commit [#2](https://github.com/energy2market/afrr-remuneration/afrr-remuneration)
- Basic README

[0.0.1]: https://github.com/energy2market/afrr-remuneration/releases/tag/v0.0.1
[0.0.2]: https://github.com/energy2market/afrr-remuneration/releases/tag/v0.0.2

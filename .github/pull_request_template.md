# Pull Request Description

> **⚠️ Important: Branch Target**
> - **New features, enhancements, and non-critical fixes**: Merge to `dev` branch
> - **Critical hotfixes only**: Merge to `main` branch (must also merge to `dev`)
> 
> Please ensure you've selected the correct base branch before submitting!

## Type of Change
<!-- Mark the appropriate option with an [x] -->
- [ ] Release (dev → main merge for production release)
- [ ] New Model Support
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Other (please describe):


## Changes Overview
<!-- Provide a brief summary of the changes in this PR -->

## Motivation and Context
<!-- Explain why this change is necessary and what problem it solves -->

## Related Issues
<!-- Link any related issues here using the syntax: Closes #123, Fixes #456 -->



----
# Conventional commit
```
type(optional scope): description
```
----
# Type candidate
  - Model Updates
    - `model`: Adding New models or Bugfix for existing models
      - ex) Add LlavaNext 
      - ex) Bugfix Whisper
  - Enhancements
    - `performance`: Optimizing some models or this library itself
      - ex) Loading RBLNModel faster
      - ex) Optimizing Memory Usage of DecoderOnlyModel
  - Code Refactor
    - `refactor`: Re-arrange class architecture, or more.
      - ex) Refactor Seq2Seq
  - Documentation
    - `doc`: Update docstring only
  - Library Dependencies
    - `dependency`: Update requirements, something like that.
  - Release
    - `release`: Merging dev to main for production release
      - ex) Release v1.2.0
  - Other
    - `other`: None of above.
      - ex) ci update
      - ex) pdm update
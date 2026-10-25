//! Deployment operations shared by the CLI and integration tests.
pub mod activation;
pub mod common;
pub mod initialize;
pub mod jobs;
pub mod launcher;
pub mod offline;
pub mod package;
pub mod retention;
pub mod service;
#[cfg(test)]
mod tests;
pub mod validation;

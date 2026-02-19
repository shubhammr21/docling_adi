// Conventional Commits configuration for commitlint
// https://www.conventionalcommits.org/
module.exports = {
  extends: ["@commitlint/config-conventional"],
  rules: {
    // Type must be one of the following
    "type-enum": [
      2,
      "always",
      [
        "feat",     // A new feature
        "fix",      // A bug fix
        "docs",     // Documentation only changes
        "style",    // Changes that do not affect the meaning of the code
        "refactor", // A code change that neither fixes a bug nor adds a feature
        "perf",     // A code change that improves performance
        "test",     // Adding missing tests or correcting existing tests
        "build",    // Changes that affect the build system or external dependencies
        "ci",       // Changes to CI configuration files and scripts
        "chore",    // Other changes that don't modify src or test files
        "revert",   // Reverts a previous commit
      ],
    ],
    // Type is required and must be lowercase
    "type-case": [2, "always", "lower-case"],
    "type-empty": [2, "never"],
    // Subject is required
    "subject-empty": [2, "never"],
    // Subject must not end with a period
    "subject-full-stop": [2, "never", "."],
    // Subject max length
    "subject-max-length": [2, "always", 100],
    // Header max length
    "header-max-length": [2, "always", 120],
    // Body max line length
    "body-max-line-length": [2, "always", 200],
    // Footer max line length
    "footer-max-line-length": [2, "always", 200],
  },
};

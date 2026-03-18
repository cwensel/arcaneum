# Homebrew formula for Arcaneum
# Install: brew install cwensel/arcaneum/arcaneum
# This file is auto-updated by the update-homebrew workflow on each release.

class Arcaneum < Formula
  include Language::Python::Virtualenv

  desc "CLI tools for semantic and full-text search with Qdrant and MeiliSearch"
  homepage "https://github.com/cwensel/arcaneum"
  url "https://github.com/cwensel/arcaneum/releases/download/v0.2.0/arcaneum-0.2.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_create(libexec, "python3.12")
    # Install with all dependencies from PyPI. Homebrew's pip_install_and_link
    # uses --no-deps which would require 50+ resource blocks for transitive deps.
    system libexec/"bin/pip", "install", buildpath
    (bin/"arc").write_env_script libexec/"bin/arc", PATH: "#{libexec}/bin:${PATH}"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/arc --version")
  end
end

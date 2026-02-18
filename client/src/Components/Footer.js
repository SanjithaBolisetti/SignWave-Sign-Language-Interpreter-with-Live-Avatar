import React from "react";
import { Link } from "react-router-dom";

function Footer() {
  return (
    <footer
      style={{
        backgroundColor: "#FFF7F3",
        borderTop: "1px solid #F3E8E2",
        marginTop: "60px",
      }}
    >
      <div className="container py-5">
        <div className="row text-center text-md-start">

          {/* Brand */}
          <div className="col-md-4 mb-4">
            <h4 className="fw-bold">
              Sign<span style={{ color: "#FF7A59" }}>Wave</span>
            </h4>
            <p className="mt-3 text-muted">
              An intelligent Indian Sign Language platform enabling inclusive
              communication through expressive 3D sign animations.
            </p>
          </div>

          {/* Quick Links */}
          <div className="col-md-4 mb-4">
            <h6 className="fw-bold mb-3">Quick Links</h6>
            <ul className="list-unstyled">
              <li className="mb-2">
                <Link
                  to="/signwave/home"
                  className="text-decoration-none text-muted"
                >
                  Home
                </Link>
              </li>
              <li className="mb-2">
                <Link
                  to="/signwave/explore"
                  className="text-decoration-none text-muted"
                >
                  Explore Now
                </Link>
              </li>
            </ul>
          </div>

          {/* Project Details */}
          <div className="col-md-4 mb-4">
            <h6 className="fw-bold mb-3">Project Details</h6>
            <p className="text-muted mb-1">
              <strong>Institution:</strong><br />
              Saveetha Engineering College
            </p>
            <p className="text-muted">
              <strong>Developed By:</strong><br />
              Arya<br />
              Atchaya<br />
              Sanjitha
            </p>
          </div>

        </div>
      </div>

      <div
        className="text-center py-3"
        style={{
          backgroundColor: "#FFF1EB",
          borderTop: "1px solid #F3E8E2",
        }}
      >
        <small className="text-muted">
          © {new Date().getFullYear()} SignWave · All Rights Reserved
        </small>
      </div>
    </footer>
  );
}

export default Footer;

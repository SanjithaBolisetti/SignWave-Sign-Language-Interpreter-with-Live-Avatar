import React from "react";
import { Link } from "react-router-dom";
import logo from "../Assets/logo.png";

function Navbar() {
  return (
    <nav className="navbar navbar-expand-lg fixed-top navbar-light bg-white shadow-sm">
      <div className="container">
        <Link className="navbar-brand fw-bold" to="/signwave/home">
          <img src={logo} alt="SignWave" width="32" className="me-2" />
          Sign<span style={{ color: "#FF7A59" }}>Wave</span>
        </Link>

        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#nav"
        >
          <span className="navbar-toggler-icon" />
        </button>

        <div className="collapse navbar-collapse" id="nav">
          <ul className="navbar-nav ms-auto">
            <li className="nav-item">
              <Link className="nav-link" to="/signwave/home">
                Home
              </Link>
            </li>
            <li className="nav-item">
              <Link className="btn btn-primary ms-3" to="/signwave/explore">
                Explore Now
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;

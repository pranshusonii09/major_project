import React, { useState } from "react";
import "./App.css"; // Import the enhanced CSS file

function App() {
  const [formData, setFormData] = useState({
    gender: "",
    age: "",
    hypertension: "",
    heart_disease: "",
    ever_married: "",
    work_type: "",
    Residence_type: "",
    avg_glucose_level: "",
    bmi: "",
    smoking_status: "",
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    const result = await response.json();
    console.log("Response:", result);

    alert(`Predicted Result: ${result.stroke}`);
  };

  return (
    <div className="app-container">
      <h1 className="title">
        <span>Patho Predict</span>
      </h1>

      {/* Animated background lines */}
      {Array.from({ length: 20 }, (_, index) => (
        <span
          key={index}
          className={`run ${index % 2 === 0 ? "run-left" : "run-right"}`}
          style={{ animationDelay: `${index * 0.2}s` }}
        ></span>
      ))}

      <div className="form-container">
        <form onSubmit={handleSubmit}>
          {/* Gender */}
          <div className="form-group">
            <label>
              Gender:
              <select name="gender" onChange={handleChange} required>
                <option value="">Select Gender</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
            </label>
          </div>

          {/* Age */}
          <div className="form-group">
            <label>
              Age:
              <input
                type="number"
                name="age"
                onChange={handleChange}
                placeholder="Age"
                required
              />
            </label>
          </div>

          {/* Hypertension */}
          <div className="form-group">
            <label>
              Hypertension:
              <select name="hypertension" onChange={handleChange} required>
                <option value="">Select</option>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </label>
          </div>

          {/* Heart Disease */}
          <div className="form-group">
            <label>
              Heart Disease:
              <select name="heart_disease" onChange={handleChange} required>
                <option value="">Select</option>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </label>
          </div>

          {/* Ever Married */}
          <div className="form-group">
            <label>
              Ever Married:
              <select name="ever_married" onChange={handleChange} required>
                <option value="">Select</option>
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </label>
          </div>

          {/* Work Type */}
          <div className="form-group">
            <label>
              Work Type:
              <select name="work_type" onChange={handleChange} required>
                <option value="">Select</option>
                <option value="Private">Private</option>
                <option value="Self-employed">Self-employed</option>
                <option value="Govt_job">Govt Job</option>
                <option value="Children">Children</option>
                <option value="Never_worked">Never worked</option>
              </select>
            </label>
          </div>

          {/* Residence Type */}
          <div className="form-group">
            <label>
              Residence Type:
              <select name="Residence_type" onChange={handleChange} required>
                <option value="">Select</option>
                <option value="Urban">Urban</option>
                <option value="Rural">Rural</option>
              </select>
            </label>
          </div>

          {/* Average Glucose Level */}
          <div className="form-group">
            <label>
              Avg Glucose Level:
              <input
                type="number"
                name="avg_glucose_level"
                onChange={handleChange}
                placeholder="Average Glucose Level"
                step="0.01"
                required
              />
            </label>
          </div>

          {/* BMI */}
          <div className="form-group">
            <label>
              BMI:
              <input
                type="number"
                name="bmi"
                onChange={handleChange}
                placeholder="BMI"
                step="0.1"
                required
              />
            </label>
          </div>

          {/* Smoking Status */}
          <div className="form-group">
            <label>
              Smoking Status:
              <select name="smoking_status" onChange={handleChange} required>
                <option value="">Select</option>
                <option value="formerly smoked">Formerly Smoked</option>
                <option value="never smoked">Never Smoked</option>
                <option value="smokes">Smokes</option>
              </select>
            </label>
          </div>

          {/* Submit Button */}
          <button type="submit">Predict</button>
        </form>
      </div>
    </div>
  );
}

export default App;

import { Modal } from 'bootstrap/dist/js/bootstrap.bundle.min';
import React, { useRef, useState } from 'react';

const colorOptions = [
  { label: 'Red', value: '#FF0000' },
  { label: 'Purple', value: '#800080' },
  { label: 'Lime', value: '#00FF00' },
  { label: 'Yellow', value: '#FFFF00' },
  { label: 'Navy', value: '#000080' },
  { label: 'Blue', value: '#0000FF' },
  { label: 'Aqua', value: '#00FFFF' },
  { label: 'Green', value: '#008000' },
  { label: 'Orange', value: '#FFA500' },
  { label: 'Pink', value: '#FFC0CB' },
  { label: 'Brown', value: '#A52A2A' },
  { label: 'Gray', value: '#808080' },
];

const ClassModel = ({ getclassdata }) => {
  const [name, setName] = useState("");
  const [color, setColor] = useState("#FF0000"); // Default color
  const modalRef = useRef(null);

  const handleSaveData = () => {
    const newName = name.trim();

    if (!newName || !color) return;

    let prevData = localStorage.getItem("Classes") || "";

    localStorage.setItem("Classes", prevData === "" ? (prevData) + `${newName}:${color}` : (prevData + ",") + `${newName}:${color}`);

    const modal = Modal.getInstance(modalRef.current);
    modal.hide();

    getclassdata(newName, color);
  }

  return (
    <div className=''>
      <div className="modal fade" ref={modalRef} data-bs-backdrop="false" id="classModel" tabIndex="-1" aria-labelledby="classModelLabel" aria-hidden="true">
        <div className="modal-dialog modal-lg"> {/* Large modal for map */}
          <div className="modal-content absolute top-16">
            <div className="modal-header">
              <h1 className="modal-title fs-5" id="classModelLabel">Create New Class</h1>
              <button type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div className="modal-body" style={{ height: '400px' }}> {/* Reduced height for map */}
              <form>
                <div className="mb-3">
                  <label htmlFor="name" className="form-label">Name</label>
                  <input
                    type="text"
                    required={true}
                    value={name}
                    className="form-control"
                    aria-describedby="emailHelp"
                    onChange={(e) => setName(e.target.value)}
                  />
                </div>
                <div className="mb-3">
                  <label htmlFor="color" className="form-label">Color</label>
                  <div className="grid grid-cols-4 gap-2">
                    {colorOptions.map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        className={`w-8 h-8 rounded-full ${color === option.value ? 'border-2 border-black' : ''}`}
                        style={{ backgroundColor: option.value }}
                        onClick={() => setColor(option.value)}
                        aria-label={option.label}
                      />
                    ))}
                  </div>
                </div>
              </form>
            </div>
            <div className="modal-footer">
              <button type="button" className="btn btn-secondary" data-bs-dismiss="modal">Close</button>
              <button type="button" className="btn btn-primary" onClick={handleSaveData}>Save changes</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ClassModel;
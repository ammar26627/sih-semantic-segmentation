import { useEffect, useState, useCallback } from "react";
import {
  MapContainer,
  TileLayer,
  LayersControl,
  ImageOverlay,
} from "react-leaflet";
import DrawControl from "./components/DrawControl";
import GeoJsonDisplay from "./components/GeoJsonDisplay";
import "leaflet/dist/leaflet.css";
import SearchComponent from "./components/SearchComponent";
import MapModel from "./components/MapModel";
import ClassModel from "./components/ClassModel";
import ScaleControl from "./components/ScaleControl";
import axios from "axios";
import "./App.css";
import styled from 'styled-components';
import closeImage from './assets/close.png'
import menuImage from './assets/menu.png'
import JSONViewer from 'json-viewer-react';


const SliderContainer = styled.div`
  background:#c3c9c8;
  padding: 10px;
  border-radius: 5px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.3);
`;

const SliderLabel = styled.label`
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
`;

const StyledSlider = styled.input`
  width: 100%;
  height:8px;
  margin-bottom: 5px;
  cursor: pointer;
  -webkit-appearance: none;
  background: #ddd;
  border-radius: 5px;
  
  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 15px;
    height: 15px;
    background: #007bff;
    border-radius: 50%;
    cursor: pointer;
  }

  &::-moz-range-thumb {
    width: 15px;
    height: 15px;
    background: #007bff;
    border-radius: 50%;
    cursor: pointer;
  }

  &::-ms-thumb {
    width: 15px;
    height: 15px;
    background: #007bff;
    border-radius: 50%;
    cursor: pointer;
  }
`;






const App = () => {
  const [imageUrl, setImageUrl] = useState(null);
  const [imageData, setImageData] = useState({});
  const [showMask, setShowMask] = useState(null);
  const [imageBounds, setImageBounds] = useState(null);
  const [showImage, setShowImage] = useState(false);
  const [ROIdata, setROIdata] = useState([]);
  const [classdata, setclassdata] = useState([]);
  const [enableClasses, setenableClasses] = useState(false);
  const [drawControl, setdrawControl] = useState(false);
  const [ROISelection, setROISelection] = useState(null);
  const [classSelection, setclassSelection] = useState(null);
  const [geoJsonData, setGeoJsonData] = useState([]);
  const [bandValues, setBandValues] = useState({
    band1: "B7",
    band2: "B4",
    band3: "B2",
  });
  const [modelSelection, setModelSelection] = useState(null);
  const [isDraggingSlider, setIsDraggingSlider] = useState(false);
  const [sliderOpen, setSliderOpen] = useState(false);
  const [modelThresHold, setModelThresHold] = useState(null);
  const [ThresholdClass, setThresholdClass] = useState([]);
  const [opacitySlider, setOpacitySlider] = useState(false);

  const handleSliderChange = (name) => (e) => {
    setImageData(prev => ({
      ...prev,
      [name]: {
        ...prev[name],
        opacity: parseFloat(e.target.value),
      },
    }));
  };

  const handleMouseDown = () => {
    setIsDraggingSlider(true);
  };

  const handleMouseUp = () => {
    setIsDraggingSlider(false);
  };
  const handleBandChange = (e, bandKey) => {
    setBandValues((prev) => ({
      ...prev,
      [bandKey]: e.target.value,
    }));
  };
  const handleModelChange = (e) => {
    setModelSelection(e.target.value);
  };
  const handleROISelection = (e) => {
    const value = e.target.value;
    setROISelection(value);

    if (value === "-1") {
      setenableClasses(false);
      setdrawControl(false);
    } else {
      setdrawControl(true);
    }
  };

  const handleClassSelection = (e) => {
    const value = e.target.value;
    setclassSelection(value);

    if (value === "-1") {
      setdrawControl(false);
    } else {
      setdrawControl(true);
    }
  };

  const getROIdata = (name = "") => {
    if (name) {
      const newData = Object.keys(localStorage)
        .filter((key) => key !== "ROI" && key !== "Classes")
        .map((key, index) => (
          <option key={index} value={localStorage.getItem(key)}>
            {key}
          </option>
        ));

      setROIdata(newData);
      setROISelection(localStorage.getItem(name));

      const value = ROISelection;

      if (value === "-1") {
        setenableClasses(false);
        setdrawControl(false);
      } else {
        setdrawControl(true);
      }
    } else {
      const newData = Object.keys(localStorage)
        .filter((key) => key !== "ROI" && key !== "Classes")
        .map((key, index) => (
          <option key={index} value={localStorage.getItem(key)}>
            {key}
          </option>
        ));

      setROIdata(newData);
    }
  };
  const getclassdata = (name = "") => {
    if (name) {
      const newData = localStorage
        .getItem("Classes")
        ?.split(",")
        .map((ele, index) => (
          <option key={index} value={ele}>
            {ele.split(":")[0]}
          </option>
        ));

      setclassdata(newData);
      setclassSelection(name);

      const value = classSelection;

      if (value === "-1") {
        setdrawControl(false);
      } else {
        setdrawControl(true);
      }
    } else {
      const newData = localStorage
        .getItem("Classes")
        ?.split(",")
        .map((ele, index) => (
          <option key={index} value={ele}>
            {ele.split(":")[0]}
          </option>
        ));

      setclassdata(newData);
    }
  };

  useEffect(() => {
    getROIdata();
    getclassdata();
  }, []);

  const generateImageFromPixels = useCallback((imageURLFromBackend) => {
    setImageUrl(imageURLFromBackend);
    setGeoJsonData([])

  }, []);
  const generateMaskFromPixels = (data) => {
    let images = {};

    Object.keys(data).forEach(key => {

      const [base64Image, opacity, area] = data[key];
      images[key] = {
        url: `data:image/png;base64,${base64Image}`,
        opacity: opacity,
        area: area,
      };
    });
    setImageData(images);
  };

  const handleSelectionClick = (bounds) => {
    setImageBounds(bounds);
  };

  const handleImageShow = () => {
    setShowImage((prev) => !prev);
  };
  let requestImage = false;
  const sendGeoJsonData = async () => {
    if (requestImage) {
      handleImageShow();
      console.log("Request already sent, skipping...");
      return; // Prevent the request from being sent again
    }
      try {
        handleImageShow();
        const combinedData = {
          "geojson": geoJsonData,
          "bands": bandValues,
          // model: modelSelection,
        };
        console.log("combinedData", combinedData);

        const response = await axios.post(
          "http://13.127.165.63:5000/get_gee_image",
          combinedData,
          {
            headers: {
              "Content-Type": "application/json",
            },
            responseType: "blob",
          }
        );
        const pixelData = await response.data;
        console.log("pixelData", pixelData);
        const imageURLFromBackend = URL.createObjectURL(pixelData);
        generateImageFromPixels(imageURLFromBackend);
        requestImage = true
      } catch (error) {
        console.error("Error sending GeoJSON data:", error);
      }
    
  };
  const handleMaskShow = () => {
    setShowMask((prev) => !prev);
  };
  let requestMask = false;
  const sendMaskData = async () => {
    if (requestMask) {
      handleMaskShow();
      console.log("Request already sent, skipping...");
      return; 
    }
      try {
        handleMaskShow();
        const combinedData = {
          "geojson": geoJsonData, 
          "model": modelSelection,
          "thresholds": { 'Urban': 3, 'Forest': 5}
        };
        console.log("mask", combinedData);

        const response = await axios.post("http://13.127.165.63:5000/get_mask", combinedData, {
          headers: {
            "Content-Type": "application/json",
          },
          responseType: "blob",
        });
        const blob = response.data;
        const reader = new FileReader();

        // Read the Blob as text
        reader.onloadend = async () => {
          const jsonData = reader.result; // This is the text content of the Blob
          const maskData = JSON.parse(jsonData); // Parse the JSON string
          console.log("Parsed Mask Data:", maskData);
          // const maskURLFromBackend = URL.createObjectURL(maskData); 
          generateMaskFromPixels(maskData);
        }
        reader.readAsText(blob);
        requestMask = true;// Load image masks after sending data
      } catch (error) {
        console.error("Error sending GeoJSON data:", error);
      }
    
  };
  useEffect(() => {

    if (classSelection) {
      setThresholdClass(prev => [...prev, classSelection.split(":")[0]]);
    }
  }, [classSelection]);

  const [settingsSelected, setSettingsSelected] = useState(true);
  const toggleOpacity = () => {
    setOpacitySlider(!opacitySlider);
  };
  return (
    <div className="relative" style={{ zIndex: "10" }}>
      <div className={`${sliderOpen ? "w-[40%]" : "w-0"} z-[1001] d-flex flex-column align-items-center sidebar position-absolute right-0 rounded-xl`} style={{ overflowY: "scroll", height: '100vh', overflowX: "hidden" }}>

        <div className="w-100 d-flex justify-between">
          <div className={`${settingsSelected ? "bg-primary cursor-pointer text-white" : "bg-light cursor-pointer text-black"} fw-bold text-lg w-[50%] text-center d-flex justify-content-center gap-3 align-items-center`} onClick={() => setSettingsSelected(true)}>
            <img src="../public/maps.gif" alt="" className="w-8 rounded-full" />Advanced Settings
          </div>

          <div className={`${!settingsSelected ? "bg-primary text-white cursor-pointer" : "bg-light cursor-pointer text-black"} fw-bold text-lg w-[50%] text-center d-flex justify-content-center gap-3 align-items-center`} onClick={() => setSettingsSelected(false)}>
            <img src="../public/maps.gif" alt="" className="w-8 rounded-full" />Geo Json
          </div>

          <img src={closeImage} alt="" width={15} className="m-3" onClick={() => setSliderOpen(false)} />
        </div>
        <div className={`${!sliderOpen ? "d-none" : "d-flex"} flex-column w-100 `}>
          {settingsSelected && <>
            <div className="rounded-lg">

              <h1 className="text-xl pt-4 pl-5 cursor-pointer">Bands</h1>
              <div className="">
                <div className="p-2 bg-white m-3 w-100">
                  <select
                    className="form-select cursor-pointer"
                    aria-label="B options"
                    style={{ maxHeight: "150px", overflowY: "auto", width: "90%" }}
                    onChange={(e) => handleBandChange(e, "band1")}
                  >
                    <option value="-1">Band 1</option>
                    {[...Array(15)].map((_, index) => (
                      <option key={index} value={`B${index + 1}`}>
                        B{index + 1}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="p-2 bg-white m-3 w-100">
                  <select
                    className="form-select cursor-pointer"
                    aria-label="B options"
                    style={{ maxHeight: "150px", overflowY: "auto", width: "90%" }}
                    onChange={(e) => handleBandChange(e, "band2")}
                  >
                    <option value="-1">Band 2</option>
                    {[...Array(15)].map((_, index) => (
                      <option key={index} value={`B${index + 1}`}>
                        B{index + 1}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="p-2 bg-white m-3 w-100">
                  <select
                    className="form-select cursor-pointer"
                    aria-label="B options"
                    style={{ maxHeight: "150px", overflowY: "auto", width: "90%" }}
                    onChange={(e) => handleBandChange(e, "band3")}
                  >
                    <option value="-1">Band 3</option>
                    {[...Array(15)].map((_, index) => (
                      <option key={index} value={`B${index + 1}`}>
                        B{index + 1}
                      </option>
                    ))}
                  </select>
                </div>

              </div>

            </div>

            <div className="rounded-lg">
              <h1 className="text-xl pt-4 pl-5">Model</h1>
              <div className="p-2 bg-white m-3">
                <select
                  className="form-select w-100 cursor-pointer"
                  aria-label="B options"
                  style={{ maxHeight: "150px", overflowY: "auto" }}
                  onChange={handleModelChange}
                >
                  <option value="-1">Model CLassifier</option>

                  <option value="Random Forest Classifier">
                    Random Forest Classifier
                  </option>
                  <option value="Parallelepiped Classifier">
                    Parallelepiped Classifier
                  </option>
                  <option value="Maximum Likelyhood Classifier">
                    Maximum Likelyhood Classifier
                  </option>
                  <option defaultValue="Mahalanobis Distance Classifier">
                    Mahalanobis Distance Classifier
                  </option>
                </select>
              </div>
            </div>
            <div className="rounded-lg">
              <h1 className="text-xl pt-4 pl-5">Threshold</h1>
              <div className="p-2 bg-white m-3">
                <input
                  type="text"
                  className="form-control"
                  placeholder="Enter threshold value"
                  onChange={(e) => {
                    const value = e.target.value;
                    if (isNaN(value)) {
                      alert("Please enter a numeric value");
                    } else {
                      setModelThresHold(value);
                    }
                  }}
                />
              </div>
            </div>

            {ThresholdClass.map((names, index) => (
              <div key={index} className="rounded-lg mb-5">
                <h1 className="text-xl pt-4 pl-5">Threshold for {names} </h1>
                <div className="p-2 bg-white m-3">
                  <input
                    type="text"
                    className="form-control"
                    placeholder={`Enter threshold for ${classSelection.split(":")[0]}`}
                    onChange={(e) => {
                      const value = e.target.value;
                      if (isNaN(value)) {
                        alert("Please enter a numeric value");
                      } else {
                        setModelThresHold((prev) => ({
                          ...prev,
                          [classSelection.split(":")[0]]: value,
                        }));
                      }
                    }}
                  />
                </div>
              </div>
            ))}
          </>}

          {!settingsSelected && (
            <div className="p-10">
              {geoJsonData ? <JSONViewer json={geoJsonData} /> : 'No data available'}
            </div>
          )}


        </div>
      </div>
      <div className="absolute z-[1000] bottom-0 flex">
        <div className="p-2">
          <button
            type="button"
            onClick={sendGeoJsonData}
            className="cursor-pointer p-1 px-3 d-flex gap-3 align-items-center text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 shadow-lg shadow-blue-500/50 dark:shadow-lg dark:shadow-blue-800/80 font-medium rounded-lg text-sm border border-black"
          >
            <img src="../public/maps.gif" alt="" className="w-6  rounded-full" />
            <div className="text-sm">Image</div>
          </button>
        </div>
        <div className="p-2">
          <button
            type="button"
            onClick={sendMaskData}
            className="cursor-pointer p-1 px-3 d-flex gap-3 align-items-center text-white bg-gradient-to-r from-blue-500 via-blue-600 to-blue-700 hover:bg-gradient-to-br focus:ring-4 focus:outline-none focus:ring-blue-300 dark:focus:ring-blue-800 shadow-lg shadow-blue-500/50 dark:shadow-lg dark:shadow-blue-800/80 font-medium rounded-lg text-sm border border-black"
          >
            <img src="../public/maps.gif" alt="" className="w-6 rounded-full" />
            <div className="text-sm">Segment</div>
          </button>
        </div>

      </div>



      <div className=" mt-3 d-flex flex-wrap gap-16 justify-content items-center absolute selection-dropdowns">
      
      </div>
      <div onClick={() => setSliderOpen(true)} className={`${sliderOpen ? 'd-none' : 'd-block'} position-absolute bg-white border rounded p-2 cursor-pointer`} style={{ zIndex: "1000", top: "20px", right: "25px" }} >
        <img src={menuImage} alt="..." width={20} />
      </div>
      <MapContainer
        center={[28.6139, 77.209]}
        zoom={4}
        dragging={!isDraggingSlider}
        style={{ height: "100vh", width: "100%", zIndex: "1" }}
      >
        <LayersControl position="bottomright">
          <LayersControl.BaseLayer name="Simple Map">
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution="&copy; OpenStreetMap contributors"
            />
          </LayersControl.BaseLayer>

          <LayersControl.BaseLayer checked name="Satellite Map">
            <TileLayer
              url="https://api.maptiler.com/maps/hybrid/{z}/{x}/{y}.jpg?key=8NCbDIVlh8Z9HQk86tuN"
              attribution='&copy; <a href="https://www.maptiler.com/copyright/">MapTiler</a> &copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
          </LayersControl.BaseLayer>
        </LayersControl>
        <div className="absolute m-3 d-flex gap-11 flex-wrap z-[1000] left-16 ">
          <SearchComponent />
          <div
            className=" d-flex gap-3"
            style={{ zIndex: "1000", top: "10px", minWidth: '280px' }}
          >
            <select
              className="form-select w-100 border p-2 rounded border-black w-full"
              aria-label="Default select example"
              onChange={handleROISelection}
              value={ROISelection || ""}
              style={{ fontSize: '12px', fontWeight: "bold" }}
            >
              <option value="-1">Region of Interest</option>
              {ROIdata}
            </select>
            <div
              className="w-50"
              style={{ zIndex: "2000", top: "50px", left: "10%", right: "10%" }}
            >
              <button
                type="button"
                className="btn btn-primary w-100"
                data-bs-toggle="modal"
                data-bs-target="#exampleModal"
              >
                New
              </button>
            </div>
            <MapModel ROIdata={getROIdata} />
          </div>
          <div
            className="d-flex gap-3"
            style={{ zIndex: "1000", top: "60px", minWidth: '280px' }}
          >
            <select
              className="form-select form-select w-100 border rounded border-black w-full"
              aria-label="Default select example"
              disabled={!enableClasses}
              value={classSelection || ""}
              onChange={handleClassSelection}
              style={{ fontSize: '12px', fontWeight: "bold" }}
            >
              <option value="-1">Classes</option>
              {classdata}
            </select>
            <div
              className="w-50"
              style={{ zIndex: "2000", top: "50px", left: "10%", right: "10%" }}
            >
              <button
                disabled={!enableClasses}
                type="button"
                className="btn btn-primary w-100"
                data-bs-toggle="modal"
                data-bs-target="#classModel"
              >
                New
              </button>
            </div>
            <ClassModel getclassdata={getclassdata} />
          </div>
          <div className={showMask ? 'z-[1000] cursor-pointer w-[178px] bg-white h-9 text-center font-bold text-xs border border-black rounded-lg ' : 'hidden'}>

            <button className="w-full h-full" onClick={toggleOpacity}>
              {opacitySlider ? 'Hide Opacity' : 'Show Opacity'}
            </button>

            {opacitySlider && (

              Object.keys(imageData).map((name, index) => (
                <SliderContainer key={index} style={{ bottom: `${30 + index * 20}%` }}>
                  <SliderLabel>{name.charAt(0).toUpperCase() + name.slice(1)} Opacity</SliderLabel>
                  <StyledSlider
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={imageData[name].opacity}
                    onChange={handleSliderChange(name)}
                    onMouseDown={handleMouseDown}
                    onMouseUp={handleMouseUp}
                    onTouchStart={handleMouseDown}
                    onTouchEnd={handleMouseUp}
                  />
                  <SliderLabel>Area: {imageData[name].area} km{<sup>2</sup>}</SliderLabel>
                </SliderContainer>))
            )}
          </div>
        </div>
        {/* Dynamically render ImageOverlay and Sliders */}
        {Object.keys(imageData).map((name, index) => (
          <div key={index}>
            {imageData[name].url && imageBounds && showMask && (
              <ImageOverlay
                url={imageData[name].url}
                bounds={imageBounds}
                opacity={imageData[name].opacity}
                eventHandlers={{ click: handleMaskShow }}
              />
            )}



          </div>
        ))}

        {imageUrl && imageBounds && showImage && (
          <ImageOverlay
            url={imageUrl}
            bounds={imageBounds}
            eventHandlers={{ click: handleImageShow }}
          />
        )}
        {drawControl ? (
          <DrawControl
            onSelectionClick={handleSelectionClick}
            setGeoJsonData={setGeoJsonData}
            setdrawControl={setdrawControl}
            setenableClasses={setenableClasses}
            ROISelection={ROISelection}
            classSelection={classSelection}
            geoJsonData={geoJsonData}

          />
        ) : null}
        <GeoJsonDisplay />
        <ScaleControl />
      </MapContainer>
    </div>
  );
};

export default App;

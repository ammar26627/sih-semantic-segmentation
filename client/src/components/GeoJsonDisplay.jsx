import axios from 'axios';
import { useEffect, useState } from 'react';
import { GeoJSON } from 'react-leaflet';

const GeoJsonDisplay = () => {
    // const [geoJsonData, setGeoJsonData] = useState([]);

    // useEffect(() => {
    //     const fetchGeoJson = async () => {
    //         try {
    //             const response = await axios.get('http://localhost:5000/api/get-geojson');
    //             setGeoJsonData(response.data);
    //         } catch (error) {
    //             console.error("Error fetching GeoJSON:", error);
    //             console.log("heyey")
    //         }
    //     };

    //     fetchGeoJson();
    // }, []);

    // return (
    //     <>
    //         {geoJsonData.length > 0 ? (
    //             geoJsonData.map((geojson, index) => (
    //                 <GeoJSON key={index} data={geojson} />
    //             ))
    //         ) : (
    //             <p>No GeoJSON data available.</p>
    //         )}
    //     </>
    // );
};

export default GeoJsonDisplay;
import L from 'leaflet';
import { useMap } from 'react-leaflet';
import { useEffect } from 'react';
import 'leaflet/dist/leaflet.css'; // Ensure you have Leaflet CSS

const ScaleControl = () => {
    const map = useMap();

    useEffect(() => {
        // Create and add scale control
        const scale = L.control.scale({
            position: 'bottomright',
            maxWidth: 150,
            imperial: false,
            metric: true,
            updateWhenIdle: true
        }).addTo(map);

        // Cleanup on unmount
        return () => {
            map.removeControl(scale);
        };
    }, [map]);

    return null;
};

export default ScaleControl;

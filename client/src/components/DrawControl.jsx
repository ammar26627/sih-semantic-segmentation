import { useEffect,useState } from 'react';
import { useMap } from 'react-leaflet';
import 'leaflet-draw/dist/leaflet.draw.css';
import Leaf from 'leaflet';
import 'leaflet-draw';



const DrawControl = ({ onSelectionClick, setdrawControl, setenableClasses, ROISelection, classSelection, setGeoJsonData, geoJsonData}) => {

    const map = useMap();
    window.type = true;

    useEffect(() => {
        const drawnItems = new Leaf.FeatureGroup();
        map.addLayer(drawnItems);

        const drawControl = new Leaf.Control.Draw({
            edit: {
                featureGroup: drawnItems,
            },
            draw: {
                polygon: true,
                polyline: false,
                rectangle: true,
                marker: true,
                circle: true,
                circlemarker: false,
            },
        });

        map.addControl(drawControl);
        const onCreate = (event) => {
            const layer = event.layer;
            drawnItems.addLayer(layer);
            const geojson = layer.toGeoJSON();
            if (ROISelection && ROISelection !== "-1" && !classSelection) {
                console.log(ROISelection.split(":")[0])
                console.log(ROISelection.split(":")[1


                ])
                geojson["properties"]['roi'] = ROISelection.split(":")[0];
                geojson["properties"]['fill'] = ROISelection.split(":")[1];
                setGeoJsonData(prevData => [...prevData, geojson]);
               
                setTimeout(() => {
                    drawnItems.removeLayer(layer);
                }, 25000)

            } else if (classSelection && classSelection !== "-1" && ROISelection && ROISelection !== "-1") {
                geojson["properties"]['class'] = classSelection.split(":")[0];
                geojson["properties"]['fill'] = classSelection.split(":")[1];
                setGeoJsonData(prevData => [...prevData, geojson]);
                setTimeout(() => {
                    drawnItems.removeLayer(layer);
                }, 30000)
            }

           


            setdrawControl(false);
            setenableClasses(true);

            if (layer && onSelectionClick && ROISelection && ROISelection !== "-1" && !classSelection) {
                const bounds = layer.getBounds();
                onSelectionClick([
                    [bounds.getSouthWest().lat, bounds.getSouthWest().lng],
                    [bounds.getNorthEast().lat, bounds.getNorthEast().lng]
                ]);
            }
        };


        // const onLayerClick = (event) => {
        //     const layer = event.layer;
        //     console.log('layer', layer);
        //     console.log('onSelectionClick', onSelectionClick);
        //     if (layer && onSelectionClick) {
        //         const bounds = layer.getBounds();
        //         onSelectionClick([
        //             [bounds.getSouthWest().lat, bounds.getSouthWest().lng],
        //             [bounds.getNorthEast().lat, bounds.getNorthEast().lng]
        //         ]);
        //     }
        // };

        map.on(Leaf.Draw.Event.CREATED, onCreate);
        // drawnItems.on('click', onLayerClick);

        return () => {
            map.off(Leaf.Draw.Event.CREATED, onCreate);
            // drawnItems.off('click', onLayerClick);
            map.removeControl(drawControl);
        };
    }, [map, onSelectionClick, ROISelection, classSelection, setdrawControl, setenableClasses, geoJsonData, setGeoJsonData]);

    return null;
};

export default DrawControl;

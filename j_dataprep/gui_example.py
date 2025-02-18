import sys
import rasterio
import geopandas as gpd
from shapely.geometry import shape
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPolygonItem, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPolygonF, QPen, QBrush, QColor, QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QRectF, QPointF
import numpy as np

class PolygonItem(QGraphicsPolygonItem):
    """ Custom class for selectable polygons. """
    def __init__(self, polygon):
        super().__init__(polygon)
        self.setPen(QPen(Qt.black, 2))
        self.setBrush(QBrush(QColor(0, 255, 0, 150)))  # Green with transparency
        self.selected = False

    def toggle_selection(self):
        """ Toggle selection state """
        self.selected = not self.selected
        if self.selected:
            self.setBrush(QBrush(QColor(255, 0, 0, 150)))  # Red when selected
        else:
            self.setBrush(QBrush(QColor(0, 255, 0, 150)))  # Back to green

class TreeRemovalApp(QWidget):
    def __init__(self, raster_path, geojson_path):
        super().__init__()

        # Scene and View
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)

        # Button to remove trees
        self.remove_button = QPushButton("Remove Trees")
        self.remove_button.clicked.connect(self.remove_selected_trees)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.remove_button)
        self.setLayout(layout)

        # Load CHM Raster and Tree Polygons
        self.load_raster(raster_path)
        self.polygons = []
        self.load_polygons(geojson_path)
        self.transform = None

        # Selection Box Variables
        self.selection_rect = None
        self.start_pos = None

    def load_raster(self, raster_path):
        """ Load CHM raster as a background image """
        with rasterio.open(raster_path) as dataset:
            img_array = dataset.read(1)  # Read first band (Assuming grayscale)
            img_array = np.interp(img_array, (img_array.min(), img_array.max()), (0, 255))  # Normalize
            img_array = img_array.astype(np.uint8)

            # Get raster transform (this maps geo-coordinates to pixel indices)
            self.transform = dataset.transform

            height, width = img_array.shape
            bytes_per_line = width
            qimage = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)

            self.scene.addPixmap(pixmap)  # Add raster to scene

    def load_polygons(self, geojson_path):
        """ Load tree polygons from GeoJSON """
        gdf = gpd.read_file(geojson_path)
        inverse_transform = ~self.transform

        for _, row in gdf.iterrows():
            geom = shape(row['geometry'])  # Convert to Shapely geometry
            coords = list(geom.exterior.coords)  # Get polygon coordinates
            print(self.transform)
            # Transform coordinates from geo-coordinates to pixel coordinates
            pixel_coords = [inverse_transform * (x, y) for x, y in coords]
            print(pixel_coords)
            # Create the polygon in pixel space
            polygon = QPolygonF([QPointF(px, py) for px, py in pixel_coords])

            # Add polygon to scene
            item = PolygonItem(polygon)
            item.setFlag(QGraphicsPolygonItem.ItemIsSelectable)
            self.scene.addItem(item)
            self.polygons.append(item)

    def remove_selected_trees(self):
        """ Remove selected polygons from the scene """
        for item in self.polygons[:]:  # Copy list to avoid iteration issues
            if item.selected:
                self.scene.removeItem(item)
                self.polygons.remove(item)

    def mousePressEvent(self, event):
        """ Handle mouse click (Selection and Drag Selection) """
        if event.button() == Qt.LeftButton:
            item = self.view.itemAt(event.pos())
            if isinstance(item, PolygonItem):
                item.toggle_selection()
            else:
                # Start drag selection
                self.start_pos = event.pos()
                self.selection_rect = self.scene.addRect(QRectF(), QPen(Qt.blue, 2, Qt.DashLine))

    def mouseMoveEvent(self, event):
        """ Handle drag movement to draw selection box """
        if self.start_pos and self.selection_rect:
            rect = QRectF(self.view.mapToScene(self.start_pos), self.view.mapToScene(event.pos())).normalized()
            self.selection_rect.setRect(rect)

    def mouseReleaseEvent(self, event):
        """ Handle selection completion and highlight polygons """
        if self.selection_rect:
            rect = self.selection_rect.rect()
            self.scene.removeItem(self.selection_rect)
            self.selection_rect = None

            # Check which polygons intersect with the selection box
            for item in self.polygons:
                if rect.intersects(item.polygon().boundingRect()):
                    item.toggle_selection()

        self.start_pos = None

if __name__ == "__main__":
    raster_file = "output/CHM.TIF"   # Replace with actual raster file
    geojson_file = "output/tree_clusters.geojson"  # Replace with actual GeoJSON file

    app = QApplication(sys.argv)
    window = TreeRemovalApp(raster_file, geojson_file)
    window.show()
    sys.exit(app.exec_())

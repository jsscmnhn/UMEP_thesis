import cmd
from j_dataprep.landcover import LandCover
from j_dataprep.DEMs import Buildings, DEMS, CHM


class SolfdShell(cmd.Cmd):
    intro = 'Welcome to the SOLFD control shell. Type help or ? to list commands.\n'
    prompt = '(SOLFD) '

    def __init__(self):
        super().__init__()
        self.buildings = None

    def do_load_buildings(self, arg):
        '''Load buildings with a bbox: load_buildings min_x min_y max_x max_y'''
        try:
            min_x, min_y, max_x, max_y = map(float, arg.split())
            bbox = (min_x, min_y, max_x, max_y)
            self.buildings = Buildings(bbox)
            print("Buildings loaded.")
        except Exception as e:
            print(f"Error loading buildings: {e}")

    def do_list_buildings(self, arg):
        '''List loaded buildings'''
        if self.buildings and self.buildings.building_geometries:
            for b in self.buildings.building_geometries[:5]:  # show only first 5
                print(b)
            if len(self.buildings.building_geometries) > 5:
                print("... and more.")
        else:
            print("No buildings loaded.")

    def do_remove_building(self, arg):
        '''Remove a building by parcel ID: remove_building <parcel_id>'''
        if self.buildings:
            self.buildings.remove_buildings(arg.strip())
            print(f"Building {arg.strip()} removed.")
        else:
            print("Load buildings first.")

    def do_quit(self, arg):
        '''Exit the shell'''
        print('Exiting.')
        return True

if __name__ == '__main__':
    SolfdShell().cmdloop()
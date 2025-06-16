from __future__ import print_function
import os
import pprint
import json
from numpy import array, nan, inf

try:
    import yaml
except ImportError:
    has_yaml = False
    print('[WARN] Can not import yaml!')
else:
    has_yaml = True

class JsonObject(dict):
    """
    It can deal with json file and yaml file.
    """

    def __init__(self, infile=None, raiseErr=False):
        dict.__init__(self)

        if not infile is None:
            if os.path.isfile(infile):
                with open(infile, 'r') as fh:
                    if has_yaml and infile.endswith('.yaml'):
                        data = yaml.load(fh)
                    else:
                        ctx = fh.read()
                        try:
                            data = json.loads(ctx)
                        except:
                            data = eval(ctx)
                self.infile = infile
                self.update(data)
            elif raiseErr:
                raise IOError('%s does not exist!' % infile)
            else:
                print('File %s does not exist! Create a blank ones.' % infile)

    def toJson(self):
        return json.dumps(self, sort_keys=True, indent=4)

    if has_yaml:
        def toYaml(self):
            return yaml.dump(dict(self))

    def pprint(self, *args, **kwds):
        pprint.pprint(self, *args, **kwds)

    def writeTo(self, savename=''):
        if not savename:
            if self.infile.endswith('.yaml'):
                savefile = '.'.join(self.infile.split('.')[:-1]) + '_out.yaml'
            else:
                savefile = '.'.join(self.infile.split('.')[:-1]) + '_out.json'

        if has_yaml and savename.endswith('.yaml'):
            savedata = self.toYaml()
        else:
            savedata = self.toJson()

        with open(savename, 'w') as fout:
            fout.write(savedata)

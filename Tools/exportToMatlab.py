import argparse
from scipy import io
from Core.EK80DataContainer import EK80DataContainer
from Core.Calculation import Calculation

# python exportToMatlab.py --file ..\Data\pyEcholabEK80data.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do signal processing on ek80 data")
    parser.add_argument(
        "--file", metavar="File", type=str, help="File where data is stored"
    )

    args = parser.parse_args()

    ekdata = EK80DataContainer(args.file)
    ekcalc = Calculation(ekdata)

    range, dr = ekcalc.calcRange()
    y_mf = ekcalc.get_y_mf()
    ycq = ekcalc.get_ycq()
    ycq_pc = ekcalc.getPulseCompressedQuadrants()
    yc = ekcalc.get_yc()
    power = ekcalc.get_power()

    io.savemat(
        "exported.mat",
        {
            "y_mf": y_mf,
            "ycq": ycq,
            "ycq_pc": ycq_pc,
            "yc": yc,
            "power": power,
            "range": range,
        },
    )

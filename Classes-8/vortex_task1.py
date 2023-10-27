import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from wedge_vortex import make_directory, make_gif, WedgeVortex

# --- initial sets
output_path = "./Task1-true-proper"
make_directory(output_path)
Nx = 520  # 520
Ny = 180  # 180
u_in = 0.04
Re = 220

# --- create class
Vortex_class = WedgeVortex(Nx=Nx, Ny=Ny, u_in=u_in, Re=Re)

# --- do evolution
steps = 20000

for step in range(0, steps):
    if step == 0:
        Vortex_class.first_step()
    else:
        Vortex_class.do_one_step()

    if step < 500:
        if step % 10 == 0:
            print(f"Now I m doing {step} step.")
            vel_magnitude = Vortex_class.return_velocity_magnitude()

            # --- PLOT: VELOCITY MAGNITUDE
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.imshow(vel_magnitude, cmap='hot')
            # cax.set_clim(vmin=1, vmax=4)
            # cbar = fig.colorbar(cax, ticks=[1.0, 2.0, 3.0, 4.0], orientation='vertical')
            # plt.show()
            napis = f'{step:05d}'
            # plt.savefig(output_path + '/Velocity-magnitude-in-step-' + napis + '.png', dpi=300)
            plt.savefig(output_path + '/Velocity-magnitude-in-step-' + napis + '.png')
            plt.close(fig)

    if step % 100 == 0:
        # Vortex_class.print_analysis()

        print(f"Now I m doing {step} step.")
        vel_magnitude = Vortex_class.return_velocity_magnitude()

        # whole_velocity = Vortex_class.return_u()
        # vector_zero = [0.0, 0.0]
        # vel_magnitude = np.array([x[:] for x in [[0.0] * Ny] * Nx])
        # vel_magnitude[:, :] = whole_velocity[:, :, 1]

        # --- PLOT: VELOCITY MAGNITUDE
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.imshow(vel_magnitude, cmap='hot')
        # cax.set_clim(vmin=1, vmax=4)
        # cbar = fig.colorbar(cax, ticks=[1.0, 2.0, 3.0, 4.0], orientation='vertical')
        # plt.show()
        napis = f'{step:05d}'
        # plt.savefig(output_path + '/Velocity-magnitude-in-step-' + napis + '.png', dpi=300)
        plt.savefig(output_path + '/Velocity-magnitude-in-step-' + napis + '.png')

        plt.close(fig)

# --- make gif
name = "Velocity-magnitude"
make_gif(output_path, name)

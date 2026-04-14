import math
import numpy as np
import matplotlib

# Must run before pyplot is imported; late use('Agg') inside functions is ignored and breaks
# file-based animation on macOS (blank MP4 / white frames).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3


def _clear_axes3d_artists(ax):
    """Matplotlib >=3.8 makes ax.lines / ax.collections read-only; remove artists explicitly."""
    for line in list(ax.lines):
        line.remove()
    for coll in list(ax.collections):
        coll.remove()


COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def plot_2d_pose(pose, pose_tree, class_type, save_path=None, excluded_joints=None):
    def init():
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(class_type)

    fig = plt.figure()
    init()
    data = np.array(pose, dtype=float)

    if excluded_joints is None:
        plt.scatter(data[:, 0], data[:, 1], color='b', marker='h', s=15)
    else:
        plot_joints = [i for i in range(data.shape[1]) if i not in excluded_joints]
        plt.scatter(data[plot_joints, 0], data[plot_joints, 1], color='b', marker='h', s=15)

    for idx1, idx2 in pose_tree:
        plt.plot([data[idx1, 0], data[idx2, 0]],
                [data[idx1, 1], data[idx2, 1]], color='r', linewidth=2.0)

    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


# def draw_pose_from_cords(img_mat_size, pose_2d, kinematic_tree, radius=2):
#     img = np.zeros(shape=img_mat_size + (3,), dtype=np.uint8)
#     lw = 2
#     pose = pose_2d.astype(np.int32)
#     for i, (idx1, idx2) in enumerate(kinematic_tree):
#         cv2.line(img, (pose[idx1, 0], pose[idx1, 1]), (pose[idx2, 0], pose[idx2, 1]), (255, 255, 255), lw)
#
#     for i, uv in enumerate(pose_2d):
#         point = tuple(uv.astype(np.int32))
#         cv2.circle(img, point, radius, COLORS[i % len(COLORS)], -1)
#     return img

def plot_3d_pose_v2(savePath, kinematic_tree, joints, title=None):
    figure = plt.figure()
    # ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), projection='3d')
    ax = Axes3D(figure)
#     ax.set_ylim(-1, 1)
#     ax.set_xlim(-1, 1)
#     ax.set_zlim(-1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=110, azim=90)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='black')
    colors = ['red', 'magenta', 'black', 'magenta', 'black', 'green', 'blue']
    for chain, color in zip(kinematic_tree, colors):
        ax.plot3D(joints[chain, 0], joints[chain, 1], joints[chain, 2], linewidth=2.0, color=color)
#     ax.set_aspect(1)
# #     plt.axis('off')
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])
#     plt.savefig(savePath)
    plt.show()

def plot_3d_motion_v2(motion, kinematic_tree, save_path, interval=50, dataset=None, title=None):
#     matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim(0, 800)
        ax.set_xlim(0, 800)
        ax.set_zlim(0, 5000)
        # ax.set_ylim(-0.75, 0.75)
        # ax.set_xlim(-0.75, 0.75)
        # ax.set_zlim(-0.75, 0.75)
        if title is not None:
            ax.set_title(title)

    motion = motion.reshape(motion.shape[0], -1, 3)
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init()

    data = np.array(motion, dtype=float)
    colors = ['red', 'magenta', 'black', 'green', 'blue','red', 'magenta', 'black', 'green', 'blue']
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        _clear_axes3d_artists(ax)
        ax.view_init(elev=110, azim=-90)
        ax.scatter(motion[index, :, 0], motion[index, :, 1], motion[index, :, 2], color='black')
        for chain, color in zip(kinematic_tree, colors):
            ax.plot3D(motion[index, chain, 0], motion[index, chain, 1], motion[index, chain, 2], linewidth=2.0, color=color)
#         ax.set_aspect('equal')
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=True, repeat_delay=50)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()


# radius = 10*offsets
def plot_3d_motion_kit(save_path, kinematic_tree, joints, title, figsize=(5, 5), interval=100, radius=246 * 12):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        # print(title)
        fig.suptitle(title)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'magenta', 'black', 'green', 'blue', 'red', 'magenta', 'black', 'green', 'blue']
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        _clear_axes3d_artists(ax)
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
        ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], color='black')
        for chain, color in zip(kinematic_tree, colors):
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=2.0, color=color)
        #         print(trajec[:index, 0].shape)
        if index > 1:
            ax.plot3D(trajec[:index, 0], np.zeros_like(trajec[:index, 0]), trajec[:index, 1], linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=True, repeat_delay=50)

    ani.save(save_path, writer='pillow')
    plt.close()

def plot_3d_motion_gt_pred(save_path, kinematic_tree, gt_joints, pred_joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        for ax in axs:
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([0, radius])
            ax.set_zlim3d([0, radius])
            fig.suptitle(title, fontsize=20)
            ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz, ax):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    def update(index):
        for i, ax in enumerate(axs):
            _clear_axes3d_artists(ax)
            ax.view_init(elev=120, azim=-90)
            ax.dist = 7.5

            MINS = motions_min[i]
            MAXS = motions_max[i]
            trajec = motions_traj[i]
            data = motions_data[i]

            plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                        MAXS[2] - trajec[index, 1], ax)

            if index > 1:
                ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                        trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                        color='blue')

            for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
                if i < 5:
                    linewidth = 4.0
                else:
                    linewidth = 2.0
                ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                        color=color)

            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

    # (seq_len, joints_num, 3)
    
    motions_data = []
    motions_traj = []
    motions_min = []
    motions_max = []
    colors = ['red', 'blue', 'black', 'red', 'blue',
                'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
                'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    for i, joints in enumerate((gt_joints, pred_joints)):
        data = joints.copy().reshape(len(joints), -1, 3)
        frame_number = data.shape[0]

        MINS = data.min(axis=0).min(axis=0)
        motions_min.append(MINS)
        MAXS = data.max(axis=0).max(axis=0)
        motions_max.append(MAXS)
        height_offset = MINS[1]

        data[:, :, 1] -= height_offset
        trajec = data[:, 0, [0, 2]]
        motions_traj.append(trajec)

        data[..., 0] -= data[:, 0:1, 0]
        data[..., 2] -= data[:, 0:1, 2]
        motions_data.append(data)

    axs = []
    fig = plt.figure(figsize=(20,10))
    axs.append(fig.add_subplot(1, 2, 1, projection='3d'))
    axs.append(fig.add_subplot(1, 2, 2, projection='3d'))
    init()

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    plt.close()

def _style_axes3d_clean(ax):
    """Hide 3D chrome without set_axis_off(), which breaks 3D line rendering in several MPL versions."""
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')


def _mocap_yup_to_mpl_xyz(p):
    """Map HumanML3D-style coords (X, Y-up, Z) to matplotlib-friendly (X, Z, Y) so mpl-Z is world-up on screen.

    Without this, mplot3d treats its Z axis as the visual “up” poorly vs data Y-up → tilted / floating look.
    """
    q = np.empty_like(p, dtype=np.float64)
    q[..., 0] = p[..., 0]
    q[..., 1] = p[..., 2]
    q[..., 2] = p[..., 1]
    return q


def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    data = np.asarray(joints, dtype=np.float64).reshape(len(joints), -1, 3)
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    MINS = data.min(axis=0).min(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    data_d = _mocap_yup_to_mpl_xyz(data)

    fig = plt.figure(figsize=figsize, facecolor='white')
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(title, fontsize=14, y=0.98)
    ax.set_facecolor('white')
    _style_axes3d_clean(ax)

    def plot_ground_plane(px0, px1, py0, py1):
        """Ground in display space: horizontal plane mpl-Z == 0 (feet / floor after Y-offset)."""
        verts = [
            [px0, py0, 0.0],
            [px0, py1, 0.0],
            [px1, py1, 0.0],
            [px1, py0, 0.0],
        ]
        g = Poly3DCollection([verts])
        g.set_facecolor((0.88, 0.88, 0.9, 0.35))
        g.set_edgecolor((0.65, 0.65, 0.7))
        g.set_linewidth(0.25)
        ax.add_collection3d(g)

    def bounds_for_frame(index):
        """Display-space bounds; keep mpl-Z (world height) anchored at ground z=0."""
        j = data_d[index]
        xs = [j[:, 0]]
        ys = [j[:, 1]]
        zs = [j[:, 2]]
        tx = trajec[: index + 1, 0] - trajec[index, 0]
        tz = trajec[: index + 1, 1] - trajec[index, 1]
        xs.append(tx)
        ys.append(tz)
        zs.append(np.zeros(index + 1, dtype=np.float64))
        xa = np.concatenate(xs)
        ya = np.concatenate(ys)
        za = np.concatenate(zs)
        dmin = np.array([xa.min(), ya.min(), za.min()])
        dmax = np.array([xa.max(), ya.max(), za.max()])
        span = np.maximum(dmax - dmin, 0.35)
        pad = span * 0.16 + 0.04
        xlim = (float(dmin[0] - pad[0]), float(dmax[0] + pad[0]))
        ylim = (float(dmin[1] - pad[1]), float(dmax[1] + pad[1]))
        zmax = float(dmax[2] + pad[2])
        zmin = float(min(0.0, dmin[2] - pad[2]))
        zlim = (zmin, zmax)
        return xlim, ylim, zlim

    def apply_lims(xlim, ylim, zlim):
        ax.autoscale(False)
        ax.set_xlim3d(xlim)
        ax.set_ylim3d(ylim)
        ax.set_zlim3d(zlim)
        try:
            ax.set_box_aspect(
                (
                    xlim[1] - xlim[0],
                    ylim[1] - ylim[0],
                    zlim[1] - zlim[0],
                )
            )
        except Exception:
            pass

    def draw_frame(index):
        _clear_axes3d_artists(ax)
        xlim, ylim, zlim = bounds_for_frame(index)
        apply_lims(xlim, ylim, zlim)
        # Oblique forward-ish view; world-up is mpl-Z after remap — reads as upright walk, not “lying in the box”.
        ax.view_init(elev=22, azim=-52)
        if hasattr(ax, "dist"):
            ax.dist = 7.0

        plot_ground_plane(xlim[0], xlim[1], ylim[0], ylim[1])

        if index > 1:
            tx = trajec[:index, 0] - trajec[index, 0]
            ty = trajec[:index, 1] - trajec[index, 1]
            ax.plot3D(
                tx,
                ty,
                np.zeros_like(tx),
                linewidth=2.0,
                color='royalblue',
            )

        pts = data_d[index]
        ax.scatter3D(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c='black', s=18, depthshade=False,
        )

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            lw = 4.0 if i < 5 else 2.0
            ax.plot3D(
                data_d[index, chain, 0],
                data_d[index, chain, 1],
                data_d[index, chain, 2],
                linewidth=lw,
                color=color,
            )

        _style_axes3d_clean(ax)

    # Still image for quick sanity check (open if MP4 looks empty in your player)
    preview_path = save_path.rsplit('.', 1)[0] + '_frame0.png'
    draw_frame(0)
    fig.savefig(preview_path, dpi=120, facecolor='white', bbox_inches='tight')

    # FuncAnimation + 3D + ffmpeg is flaky on some stacks; explicit grab_frame on Agg is reliable.
    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='MoMask'), bitrate=1800)
        with writer.saving(fig, save_path, dpi=100):
            for fi in range(frame_number):
                draw_frame(fi)
                writer.grab_frame()
    except (RuntimeError, ValueError, FileNotFoundError) as err:
        import warnings
        base = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        gif_path = base + '_fallback.gif'
        pw = PillowWriter(fps=fps)
        with pw.saving(fig, gif_path, dpi=80):
            for fi in range(frame_number):
                draw_frame(fi)
                pw.grab_frame()
        warnings.warn('FFmpeg path failed (%r); wrote GIF instead: %s' % (err, gif_path))
    plt.close()


def plot_3d_motion_old(motion, pose_tree, class_type, save_path, interval=300, excluded_joints=None):
    matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_ylim(-0.75, 0.75)
        ax.set_xlim(-0.75, 0.75)
        ax.set_zlim(-0.75, 0.75)
        # ax.set_ylim(-1.0, 0.2)
        # ax.set_xlim(-0.2, 1.0)
        # ax.set_zlim(-1.0, 0.4)
        ax.set_title(class_type)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init()

    data = np.array(motion, dtype=float)
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        _clear_axes3d_artists(ax)
        if excluded_joints is None:
            ax.scatter(data[index, :, 0], data[index, :, 1], data[index, :, 2], color='b', marker='h', s=15)
        else:
            plot_joints = [i for i in range(data.shape[1]) if i not in excluded_joints]
            ax.scatter(data[index, plot_joints, 0], data[index, plot_joints, 1], data[index, plot_joints, 2], color='b', marker='h', s=15)

        for idx1, idx2 in pose_tree:
            ax.plot([data[index, idx1, 0], data[index, idx2, 0]],
                    [data[index, idx1, 1], data[index, idx2, 1]], [data[index, idx1, 2], data[index, idx2, 2]], color='r', linewidth=2.0)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()


def plot_2d_motion(motion, pose_tree, axis_0, axis_1, class_type, save_path, interval=300):
    matplotlib.use('Agg')

    fig = plt.figure()
    plt.title(class_type)
    # ax = fig.add_subplot(111, projection='3d')
    data = np.array(motion, dtype=float)
    frame_number = data.shape[0]
    # dim (frame, joints, xyz)
    print(data.shape)

    def update(index):
        plt.clf()
        plt.xlim(-0.7, 0.7)
        plt.ylim(-0.7, 0.7)
        plt.scatter(data[index, :, axis_0], data[index, :, axis_1], color='b', marker='h', s=15)
        for idx1, idx2 in pose_tree:
            plt.plot([data[index, idx1, axis_0], data[index, idx2, axis_0]],
                    [data[index, idx1, axis_1], data[index, idx2, axis_1]], color='r', linewidth=2.0)

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()

def plot_3d_multi_motion(motion_list, kinematic_tree, save_path, interval=50, dataset=None):
    matplotlib.use('Agg')

    def init():
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if dataset == "mocap":
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlim(0, 3)
            ax.set_zlim(-1.5, 1.5)
        else:
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)
        # ax.set_ylim(-1.0, 0.2)
        # ax.set_xlim(-0.2, 1.0)
        # ax.set_zlim(-1.0, 0.4)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = p3.Axes3D(fig)
    init()

    colors = ['red', 'magenta', 'black', 'magenta', 'black', 'green', 'blue']
    frame_number = motion_list[0].shape[0]
    # dim (frame, joints, xyz)
    # print(data.shape)
    print("Number of motions %d" % (len(motion_list)))
    def update(index):
        _clear_axes3d_artists(ax)
        if dataset == "mocap":
            ax.view_init(elev=110, azim=-90)
        else:
            ax.view_init(elev=110, azim=90)
        for motion in motion_list:
            for chain, color in zip(kinematic_tree, colors):
                ax.plot3D(motion[index, chain, 0], motion[index, chain, 1], motion[index, chain, 2],
                          linewidth=4.0, color=color)
        plt.axis('off')

#         ax.set_xticks([])
#         ax.set_yticks([])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=interval, repeat=False, repeat_delay=200)
    # update(1)
    # plt.show()
    # Writer = writers['ffmpeg']
    # writer = Writer(fps=15, metadata={})
    ani.save(save_path, writer='pillow')
    plt.close()




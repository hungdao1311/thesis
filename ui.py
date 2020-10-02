import logging
import os
import random
import wx

import wrapper

logging.basicConfig(filename='app',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

APP_HEIGHT = 800
APP_WIDTH = 1200


class CustomConsoleHandler(logging.StreamHandler):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, textctrl):
        """"""
        logging.StreamHandler.__init__(self)
        self.textctrl = textctrl

    # ----------------------------------------------------------------------
    def emit(self, record):
        """Constructor"""
        formatter = logging.Formatter('%(asctime)s  -  %(levelname)s    %(message)s')
        self.setFormatter(formatter)
        msg = self.format(record)
        self.textctrl.WriteText(msg + "\n")
        self.flush()


class LoggerShowing(wx.TextCtrl):
    def __init__(self, logger, parent=None):
        super(LoggerShowing, self).__init__(parent, size=(-1, 200),
                                            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL)
        self.logger = logger
        txtHandler = CustomConsoleHandler(self)
        self.logger.addHandler(txtHandler)
        self.logger.info('Open app successfully')


class ImagePanel(wx.Panel):
    def __init__(self, parent=None):
        wx.Panel.__init__(self, parent)
        # png = wx.EmptyBitmap()
        self.imageCtrl = wx.StaticBitmap(self, wx.ID_ANY, wx.EmptyBitmap(APP_WIDTH, APP_HEIGHT))
        # wx.StaticBitmap(self, -1, png, ((APP_WIDTH - png.GetWidth())/2, (APP_HEIGHT - png.GetWidth())/2), (png.GetWidth(), png.GetHeight()))

    def set_image(self, image_path):
        if not os.path.exists(image_path):
            return
        wx.BeginBusyCursor()
        self.Freeze()
        png = wx.Image(image_path, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        pos = (int((APP_WIDTH - png.GetWidth()) / 2), int((APP_HEIGHT - 200 - png.GetHeight()) / 2))
        self.imageCtrl = wx.StaticBitmap(self, wx.ID_ANY, png, pos=pos, size=(png.GetWidth(), png.GetHeight()))
        self.Thaw()
        wx.EndBusyCursor()


class MainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, parent=None, title='People counting', size=(APP_WIDTH, APP_HEIGHT))
        main_panel = wx.Panel(self, size=(APP_WIDTH, APP_HEIGHT))  # main panel

        self.ori_image = None
        self.logger = logging.getLogger("app")

        splitter_window = wx.SplitterWindow(main_panel, 1, style=wx.SP_LIVE_UPDATE)
        splitter_window.SetMinimumPaneSize(550)

        # top panel
        self.image_panel = ImagePanel(splitter_window)

        # bot panel
        bot_panel = wx.Panel(splitter_window)
        bot_sizer = wx.BoxSizer(wx.HORIZONTAL)
        image2 = 'icons8-opened-folder-96.png'
        image2 = wx.Image(image2, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.open_button = wx.BitmapButton(bot_panel, id=1, bitmap=image2,
                                           size=(image2.GetWidth() + 104, image2.GetHeight() + 104))

        image3 = 'icons8-play-64.png'
        image3 = wx.Image(image3, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.running_button = wx.BitmapButton(bot_panel, id=2, bitmap=image3,
                                              size=(image3.GetWidth() + 136, image3.GetHeight() + 136))
        log_text = LoggerShowing(self.logger, parent=bot_panel)

        bot_sizer.Add(self.open_button)
        bot_sizer.Add(self.running_button)
        bot_sizer.Add(log_text, wx.EXPAND)
        bot_panel.SetSizer(bot_sizer)
        # bottom panel
        splitter_window.SplitHorizontally(self.image_panel, bot_panel, -200)
        splitter_window.SetSashGravity(0)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(splitter_window, 1, wx.EXPAND)
        main_panel.SetSizer(sizer)
        self.open_button.Bind(wx.EVT_BUTTON, self.on_open_file)
        self.running_button.Bind(wx.EVT_BUTTON, self.on_running_predict)
        self.Layout()
        self.Show()

    def on_open_file(self, evt):
        self.image_panel.set_image("/Users/hungdao/Pictures/black.png")
        with wx.FileDialog(None, "Open",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            img_path = fileDialog.GetPath()
            self.ori_image = img_path
            self.logger.info(f'Opened image')
            self.image_panel.set_image(img_path)

    def on_running_predict(self, evt):
        self.logger.info(f'Start processing image {self.ori_image}')
        self.Freeze()
        wx.BeginBusyCursor()
        res_img, count = wrapper.scan_image(self.ori_image)
        self.Thaw()
        self.logger.info('End processing')
        self.logger.info('There are ' + count + ' people(s)')
        self.image_panel.set_image(res_img)
        wx.EndBusyCursor()
        


if __name__ == "__main__":
    app = wx.App(False)
    frame = MainFrame()
    app.MainLoop()

import gi
gi.require_version('Notify', '0.7')
from gi.repository import Notify

# One time initialization of libnotify
Notify.init("My Program Name")

# Create the notification object
summary = "Wake up!"
body = "Meeting at 3PM!"
notification = Notify.Notification.new(
    summary,
    body, # Optional
)

# Actually show on screen
notification.show()

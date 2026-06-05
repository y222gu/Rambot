import tkinter as tk
from tkinter import ttk

class Subscriber:
    def __init__(self):
        self.name = "Subscriber"
        print('Subscriber initialized')

    def update(self, message):
        print("Need to implement update method in Subscriber subclass")

class Publisher:
    def __init__(self, events):
        # Maps event names to subscribers
        self.events = {event: dict() for event in events}
        print('Publisher initialized with events:', self.events.keys())

    def get_subscribers(self, event):
        return self.events[event]

    def register(self, event, who, callback=None):
        if callback is None:
            callback = getattr(who, 'update')
        self.get_subscribers(event)[who] = callback
        print(f'{who.name} registered for event "{event}"')

    def unregister(self, event, who):
        del self.get_subscribers(event)[who]
        print(f'{who.name} unregistered from event "{event}"')

    def dispatch(self, event, message):
        print(f'Dispatching event "{event}" with message: "{message}"')
        for subscriber, callback in self.get_subscribers(event).items():
            callback(message)

# New ObservableObserver class that combines Publisher and Subscriber
class ObservableObserver(Publisher, Subscriber):
    def __init__(self, events):
        Publisher.__init__(self, events)   # Initialize Publisher part
        Subscriber.__init__(self)          # Initialize Subscriber part
        self.name = "ObservableObserver"   # Give a name to this instance

    def update(self, message):
        print(f'{self.name} got message "{message}"')


# Test setup
if __name__ == "__main__":
    events = ["event1", "event2"]
    observable_observer = ObservableObserver(events)

    # ObservableObserver subscribes to its own event to simulate both roles
    observable_observer.register("event1", observable_observer)

    # ObservableObserver dispatches an event, which it should also receive
    observable_observer.dispatch("event1", "Hello, ObservableObserver!")

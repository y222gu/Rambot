import tkinter as tk
from tkinter import ttk

class Subscriber:
    def __init__(self):
        self.name = 'Subscriber'
        print('Subscriber inited')

    def handle_update_from_publsiher(self, message):
        print('{} got message "{}"'.format(self.name, message))
        
class Publisher:
    def __init__(self, events):
        # Maps event names to subscribers
        self.events = {event: dict() for event in events}
        print('Publisher initialized with events:', self.events.keys())
    def get_subscribers(self, event):
        return self.events[event]
    def add_subscriber(self, event, who, callback=None):
        if callback is None:
            callback = getattr(who, 'handle_update_from_publsiher')
        self.get_subscribers(event)[who] = callback
        print(f'{who.name} registered for event "{event}"')
    def remove_subscriber(self, event, who):
        del self.get_subscribers(event)[who]
        print(f'{who.name} unregistered from event "{event}"')
    def dispatch(self, event, *args):
        results = []
        for subscriber, callback in self.get_subscribers(event).items():
            result = callback(*args)
            # print(f'{subscriber.name} returned {result} for event "{event}"')
            if result is not None:
                results.append(result)
                # print(f'{subscriber.name} returned {result} for event "{event}"')
        return results
                
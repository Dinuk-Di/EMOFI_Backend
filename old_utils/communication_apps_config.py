COMMUNICATION_APPS_LIST = {
    "WhatsApp": {
        "process": "WhatsApp.exe",
        "deep_links": {
            "chat": "whatsapp://send?phone={phone}",
            "call": "whatsapp://call?phone={phone}"
        },
        "web_urls": {
            "chat": "https://web.whatsapp.com/send?phone={phone}",
            "call": "https://web.whatsapp.com/call?phone={phone}"
        }
    },
    "Skype": {
        "process": "Skype.exe",
        "deep_links": {
            "chat": "skype:{id}?chat",
            "call": "skype:{id}?call"
        },
        "web_urls": {
            "chat": "https://web.skype.com/",
            "call": "https://web.skype.com/"
        }
    },
    "Microsoft Teams": {
        "process": "ms-teams.exe",
        "deep_links": {
            "chat": "msteams://teams.microsoft.com/l/chat/0/0?users={id}",
            "call": "msteams://teams.microsoft.com/l/call/0/0?users={id}"
        },
        "web_urls": {
            "chat": "https://teams.microsoft.com/_#/conversations/?ctx=chat",
            "call": "https://teams.microsoft.com/_#/calls/"
        }
    },
    "Discord": {
        "process": "Discord.exe",
        "deep_links": {
            "chat": "discord://channels/@me/{id}",
            "call": "discord://call/{id}"
        },
        "web_urls": {
            "chat": "https://discord.com/channels/@me/{id}",
            "call": "https://discord.com/call/{id}"
        }
    },
    "Telegram Desktop": {
        "process": "Telegram.exe",
        "native": {
            "chat": "tg://resolve?domain={id}",
            "call": "tg://call?domain={id}"
        },
        "web": {
            "chat": "https://web.telegram.org/k/#{id}",
            "call": "https://web.telegram.org/k/#{id}&call=1"
        },
        "requirements": {
            "call": ["id"],
            "chat": ["id"]
        }
    }
}
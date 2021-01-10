#pip install -U git+https://github.com/Rapptz/discord.py@rewrite

import discord
import time
import asyncio
from ainitalk import askaini

ask = askaini()

dic = {"indraujo#4534":1}

def read_token():
    with open("token.txt","r") as f:
        lines = f.readlines()
        return lines[0].strip()

token = read_token()
serverid = 797356426585964544
messages = joined = 0

client = discord.Client()

async def update_stats():
    await client.wait_until_ready()
    global messages, joined

    while not client.is_closed():
        try:
            with open("stats.txt","a") as f:
                f.write(f"Time: {int(time.time())}, Messages: {messages}, Members Joined {joined}\n")
            messages = 0
            joined = 0
            await asyncio.sleep(5) 
        
        except Exception as e:
            print(e)
            await asyncio.sleep(5) 

@client.event
async def on_member_update(before, after):
    n = after.nick
    if n :
        if n.lower().count("tim")>0:
            last = before.nick
            if last:
                await after.edit(nick=last)
            else:
                await after.edit(nick="NO STOP THAT")

@client.event
async def on_member_join(member):
    global joined
    joined += 1
    for channel in member.server.channels:
        await client.send_message(f"""Welcome to the server {member.mention}""")

@client.event
async def on_message(message):
    global messages
    messages += 1
    
    #print(message)
    id = client.get_guild(serverid)
    channels = ["commands"]
    valid_users = ["indraujo#4534"]
    bad_words = ["bad", "stop", "45"]
    for word in bad_words:
        if message.content.count(word) >0:
            print("A bad word was said")
            await message.channel.purge(limit=1)

    if message.content == "!help":
        embed = discord.Embed(title="Help on Bot", description="Some useful commands")
        embed.add_field(name="!hello", value = "Greet the users")
        embed.add_field(name="!users", value = "Show the number of users")
        await message.channel.send(content=None, embed=embed)        


    if str(message.channel) in channels and str(message.author) in valid_users:
        print(message.content)
        if message.content.find("hi aini") != -1:
            await message.channel.send(f"""Hi {message.author.name}, do you need something dear?""")
        elif message.content == "!users":
            await message.channel.send(f"""# of Members {id.member_count}""")
        else:
            answer = ask.askaini(message.content)
            await message.channel.send(answer)
    else:
        print(f"""User: {message.author} tried to do command {message.content}, in channel {message.channel}""")

client.loop.create_task(update_stats())
client.run(token)

# https://discordapp.com/oauth2/authorize?client_id=797348008064712705&scope=bot&permissions=0

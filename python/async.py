import asyncio

async def test():
    print("test")
    return 42


async def main():
    """
    """
    task = asyncio.create_task(test())
    print("main")

    await test()

    await task


asyncio.run(main())

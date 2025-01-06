import modal

app = modal.App()

@app.function(schedule=modal.Period(days=1))
def single_tool_cron():
    pass


@app.function(schedule=modal.Period(days=1))
def multiple_tool_cron():
    pass

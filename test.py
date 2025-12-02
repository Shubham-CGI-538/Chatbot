from src.chatbot.tools.tools_utils import get_relevant_ids, get_relevant_ids_booleans
print(get_relevant_ids("NZE"))

print(get_relevant_ids_booleans(
    original_query="Subsurface wells",
    refined_query="subsurface wells OR wellbore OR borehole",
    min_should=1
))


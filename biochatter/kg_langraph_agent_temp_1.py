def generate_query_with_langgraph_agent(
    generated_query: Dict[str, str],
    connection_args: Dict,
    conversation_factory: Callable[[], GptConversation],
    prompt: str,
    question: str,
    query_language: Optional[str] = "Cypher",
):
    parser = JsonOutputToolsParser(return_id=True)
    neodriver = None
    def _log_message(msg: str):
        generated_query["logs"] = generated_query["logs"] + f"{datetime.now().isoformat()} - {msg}\n" 
    def _connect_db():
        nonlocal neodriver
        if neodriver is not None:
            return
        try:
            db_uri = "bolt://" + connection_args["host"] + \
                ":" + connection_args["port"]
            neodriver = nu.Driver(
                db_name=(
                    connection_args["db_name"] 
                    if "db_name" in connection_args else "neo4j"
                ),
                db_uri=db_uri,
            )
        except Exception as e:
            logger.error(e)   
    def _query_graph_database(query: str):
        nonlocal neodriver
        try:
            _connect_db()
            result = neodriver.query(query)
            return result
        except Exception as e:
            logger.error(str(e))
            return str(e)
    def _get_conversation(model_name: Optional[str] = None,) -> GptConversation:
        """
        Create a conversation object given a model name.

        Args:
            model_name: The name of the model to use for the conversation.

        Returns:
            A BioChatter Conversation object for connecting to the LLM.

        Todo:
            Genericise to models outside of OpenAI.
        """

        conversation = GptConversation(
            model_name=model_name or "gpt-3.5-turbo",
            prompts={},
            correct=False,
        )
        conversation.set_api_key(
            api_key=os.getenv("OPENAI_API_KEY"), user="test_user"
        )
        return conversation    
    def _query_tools(state: List[BaseMessage]) -> List[BaseMessage]:
        nonlocal parser
        tool_invocation: AIMessage = state[-1]
        parsed_tool_calls = parser.invoke(tool_invocation)
        results = []
        for parsed_call in parsed_tool_calls:
            query = parsed_call["args"]["revised_query"] if "revised_query" in parsed_call["args"] else None
            if query is not None:
                result = _query_graph_database(query)
                results.append({"query": query, "result": result[0]})
                continue
            queries = parsed_call["args"]["search_queries"]
            for query in queries:
                result = _query_graph_database(query)
                results.append({"query": query, "result": result[0]})
        if len(results) > 1:
            # if there are multiple results, we only return the first no-empty result
            for res in results:
                if len(res["result"]) > 0:
                    return ToolMessage(
                        content=json.dumps(res),
                        tool_call_id=parsed_call["id"]
                    )
        return ToolMessage(
            content=json.dumps(results[0]),
            tool_call_id=parsed_call["id"]
        )
    actor_prompt_template = ChatPromptTemplate.from_messages(
        [(
            "system",
            ("As a senior biomedical researcher and graph database expert, "
             f"your task is to generate '{query_language}' queries to extract data from our genomicKB graph database based on the user's question. " 
             """The database utilizes a knowledge graph to consolidate genomic datasets and annotations from over 30 consortia, representing genomic 
entities as nodes and relationships as edges with properties.
Current time {time}
{first_instruction}""")
        ), 
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Only generate query according to the user's question above."),]
    ).partial(
        time=lambda: datetime.now().isoformat()
    )
    conversation = conversation_factory()
    assert conversation is not None and conversation.chat is not None
    llm = conversation.chat
    initial_answer_chain = actor_prompt_template.partial(
        first_instruction=prompt,
    ) | llm.bind_tools(
        tools=[GenerateQuery],
        tool_choice="GenerateQuery",
    )
    validator = PydanticToolsParser(tools=[GenerateQuery])
    first_responder = ResponderWithRetries(
        runnable=initial_answer_chain,
        validator=validator,
    )
    first_responder.respond([HumanMessage(
        content=question
    )])

    revise_instruction = """
Revise you previous query using the query result and follow the guidelines:
1. if you consistently obtain empty result, please consider removing constraints, like relationship constraint to try to obtain some results.
2. you should use previous critique to improve your query.
"""
    revision_chain = actor_prompt_template.partial(
    first_instruction=revise_instruction,
    ) | llm.bind_tools(tools=[ReviseQuery], tool_choice="ReviseQuery")
    revision_validator = PydanticToolsParser(tools=[ReviseQuery])
    revisor = ResponderWithRetries(
        runnable=revision_chain,
        validator=revision_validator,
    )
    
    builder = MessageGraph()
    builder.add_node("generate", first_responder.respond)
    builder.add_node("query_tools", _query_tools)
    builder.add_node("revise", revisor.respond)
    builder.add_edge("generate", "query_tools")
    builder.add_edge("query_tools", "revise")
    
    builder.add_conditional_edges("revise", should_continue)
    builder.set_entry_point("generate")
    graph = builder.compile()
    
    events = graph.stream(
        [HumanMessage(content=question)], {
            "recursion_limit": 30
        }
    )
    for i, step in enumerate(events):
        node, output = next(iter(step.items()))
        _log_message(f"## {i+1}. {node}")
        try:
            _log_message(
                f'Answer: {parser.invoke(output)[0]["args"]["answer"]}')
            _log_message(
                f'Reflection | Improving: {parser.invoke(output)[0]["args"]["reflection"]["improving"]}')
            _log_message('Reflection | Search Queries:')
            
            for y, sq in enumerate(parser.invoke(output)[0]["args"]["search_queries"]):
                _log_message(f"{y+1}: {sq}")
            _log_message("✦✧✦✧✦✧✧✦✧✦✧ Node Output ✦✧✦✧✦✧✧✦✧✦✧")
            continue
    
        except Exception as e:
            logger.error(str(output)[:100] + " ...")
            _log_message(str(output)[:100] + " ...")
    
    _log_message("\n\n✦✧✦✧✦✧✧✦✧✦✧ Final Generated Response ✦✧✦✧✦✧✧✦✧✦✧\n\n")
    final_result = parser.invoke(step[END][-1])[0]["args"]["answer"]
    _log_message(final_result)
    generated_query["result"] = final_result
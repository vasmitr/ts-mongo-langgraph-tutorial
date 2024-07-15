# Technical Guide: Implementing an AI Agent with LangGraph and Vector Search in Deno/TypeScript

This guide outlines the implementation of a conversational AI agent with short-term and long-term memory capabilities using Deno/TypeScript, LangGraph, and MongoDB Atlas.

The jupyter notebook is [here](https://github.com/vasmitr/ts-mongo-langgraph-tutorial/blob/main/notebooks/ts-mongo-langgraph-tutorial.ipynb).
See how to set up [jupyter with Deno kernel](https://docs.deno.com/runtime/manual/tools/jupyter/)

## Tech Stack

- Deno: JavaScript/TypeScript runtime
- LangGraph: Framework for building stateful LLM applications
- MongoDB Atlas: Cloud database with vector search capabilities
- OpenAI Embeddings: For creating text vector representations
- Anthropic's Claude: Language model for the conversational agent

## Implementation Steps

### 1. Install dependencies

```bash
deno add npm:zod \
npm:mongodb npm:@langchain/mongodb \
npm:langchain npm:@langchain/langgraph npm:@langchain/core npm:@langchain/community \
npm:@langchain/anthropic npm:@langchain/openai
```

### 2. Environment Setup

create an `.env` file with your tokens

```bash
OPENAI_API_KEY = "";
TAVILY_API_KEY = "";
ANTHROPIC_API_KEY = "";
MONGO_URI = "";
```

Load environment, then set up Mongo client and get your history collection

```typescript
import { load } from "std/dotenv/mod.ts";
import { MongoClient } from "mongodb";

const env: Record<string, string> = await load({ envPath: "../.env" });

const { MONGO_URI } = env;

const client = new MongoClient(MONGO_URI);
const collection = client.db().collection("checkpoints");
```

### 3. Vector Search Implementation

First, create Atlas index see [langchain docs](https://js.langchain.com/v0.2/docs/integrations/vectorstores/mongodb_atlas/)

```javascript
    {
        "fields": [
            {
            "numDimensions": 1024,
            "path": "embedding",
            "similarity": "euclidean",
            "type": "vector"
            }
        ]
    }
```

then prepare embeddings, we'll use OpenAI solution:

```typescript
import { OpenAIEmbeddings } from "@langchain/openai";

const { OPENAI_API_KEY } = env;

// Initialize embeddings model
export const embeddings1024 = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
  dimensions: 1024,
  apiKey: OPENAI_API_KEY,
});
```

Set up the store

```typescript
import { MongoDBAtlasVectorSearch } from "npm:@langchain/mongodb@^0.0.4";

export const vectorStore = new MongoDBAtlasVectorSearch(embeddings1024, {
  collection: collection,
  indexName: "default",
  textKey: "messages.content",
  embeddingKey: "embedding",
});
```

Finally, add function for history retrieval

```typescript
import { z } from "zod";
import { DynamicStructuredTool } from "@langchain/core/tools";

export const getHistory = new DynamicStructuredTool({
  name: "get_history",
  description: "Use text query to perform vector search against chat history",
  schema: z.object({
    query: z.string(),
  }),
  func: async function ({ query }) {
    console.log("\x1b[33m%s\x1b[0m", `[DEBUG] history search: ${query}`);
    const embededQuery = await embeddings1024.embedQuery(query);
    const res = await vectorStore.similaritySearchVectorWithScore(
      embededQuery,
      3
    );
    const history = res
      ?.map(
        (rec: Array<Record<string, any>>) => rec?.[0]?.metadata.history || ""
      )
      .join("; ");
    return history;
  },
});
```

### 4. Adding memory to graph using MongoDB

[Example of custom checkpointer implementation with Postgres](https://langchain-ai.github.io/langgraphjs/how-tos/persistence-postgres/?h=post)

Here we extend our checkpoint with embeddings, before inserting them into mongo

```typescript
import { MongoClient, Collection } from "mongodb";

import { BaseMessage } from "@langchain/core/messages";
import { RunnableConfig } from "@langchain/core/runnables";
import { load } from "@langchain/core/load";

import {
  BaseCheckpointSaver,
  Checkpoint,
  CheckpointMetadata,
  CheckpointTuple,
} from "npm:@langchain/langgraph@0.0.26";

const { MONGO_URI } = env;

// define custom serializer for `checkpoint` and `metadata` values
const CustomSerializer = {
  stringify(obj: Record<string, unknown>) {
    return JSON.stringify(obj);
  },

  async parse(data: string) {
    return await load(data.toString());
  },
};

interface CheckpointRecord {
  checkpoint: string;
  metadata: string;
  parent_id?: string;
  thread_id: string;
  checkpoint_id: string;
  embedding: number[];
  history: string;
  timestamp: Date;
}

export class MongoSaver extends BaseCheckpointSaver {
  private client: MongoClient;
  private isSetup: boolean;
  public collection: Collection<CheckpointRecord>;

  constructor(client: MongoClient) {
    super(CustomSerializer);
    this.client = client;
    this.collection = this.client.db().collection("checkpoints");
    this.isSetup = false;
  }

  static fromConnString(connString: string = MONGO_URI || ""): MongoSaver {
    return new MongoSaver(new MongoClient(connString));
  }

  private async setup(): Promise<void> {
    if (this.isSetup) return;

    try {
      await this.collection.findOne();
      this.isSetup = true;
    } catch (error) {
      console.error("Error querying checkpoints collection", error);
      throw error;
    }
  }

  // below 3 methods are necessary for any checkpointer implementation: getTuple, list and put
  async getTuple(config: RunnableConfig): Promise<CheckpointTuple | undefined> {
    await this.setup();
    const { thread_id, checkpoint_id } = config.configurable || {};

    try {
      if (checkpoint_id) {
        const res = await this.collection.findOne({ thread_id, checkpoint_id });

        if (res) {
          const mappedRes = {
            config,
            checkpoint: (await this.serde.parse(res.checkpoint)) as Checkpoint,
            metadata: (await this.serde.parse(
              res.metadata
            )) as CheckpointMetadata,
            parentConfig: res.parent_id
              ? {
                  configurable: {
                    thread_id,
                    checkpoint_id: res.parent_id,
                  },
                }
              : undefined,
          };

          return mappedRes;
        }
      } else {
        const res = await this.collection.findOne(
          { thread_id },
          { sort: { timestamp: -1 } }
        );

        if (res) {
          const mappedRes = {
            config: {
              configurable: {
                thread_id: res.thread_id,
                checkpoint_id: res.checkpoint_id,
              },
            },
            checkpoint: (await this.serde.parse(res.checkpoint)) as Checkpoint,
            metadata: (await this.serde.parse(
              res.metadata
            )) as CheckpointMetadata,
            parentConfig: res.parent_id
              ? {
                  configurable: {
                    thread_id: res.thread_id,
                    checkpoint_id: res.parent_id,
                  },
                }
              : undefined,
          };

          return mappedRes;
        }
      }
    } catch (error) {
      console.error("Error retrieving checkpoint", error);
      throw error;
    }

    return undefined;
  }

  async *list(
    config: RunnableConfig,
    limit?: number,
    before?: RunnableConfig
  ): AsyncGenerator<CheckpointTuple> {
    await this.setup();
    const { thread_id } = config.configurable || {};
    let query: Record<string, unknown> = { thread_id };

    const params: (string | number)[] = [thread_id];
    if (before?.configurable?.checkpoint_id) {
      query = {
        ...query,
        checkpoint_id: { $lt: before.configurable.checkpoint_id },
      };
      params.push(before.configurable.checkpoint_id);
    }
    let options: Record<string, unknown> = { sort: { timestamp: -1 } };
    if (limit) {
      query.limit = params.length + 1;
      params.push(limit);
    }

    try {
      const res = await this.collection.find(query, options).toArray();

      for (const row of res) {
        yield {
          config: {
            configurable: {
              thread_id: row.thread_id,
              checkpoint_id: row.checkpoint_id,
            },
          },
          checkpoint: (await this.serde.parse(row.checkpoint)) as Checkpoint,
          metadata: (await this.serde.parse(
            row.metadata
          )) as CheckpointMetadata,
          parentConfig: row.parent_id
            ? {
                configurable: {
                  thread_id: row.thread_id,
                  checkpoint_id: row.parent_id,
                },
              }
            : undefined,
        };
      }
    } catch (error) {
      console.error("Error listing checkpoints", error);
      throw error;
    }
  }

  async put(
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata
  ): Promise<RunnableConfig> {
    await this.setup();
    try {
      const messages =
        (checkpoint?.channel_values?.messages as BaseMessage[]) || [];
      const lastMessage = messages?.[messages?.length - 1] || {};

      let text: string;

      if (lastMessage.content instanceof Array) {
        const message: Record<string, unknown> =
          lastMessage.content.find((message) => message.type === "text") || {};
        text = message?.text as string;
      } else {
        text = lastMessage.content;
      }

      if (text) {
        const embeddings = await embeddings1024.embedDocuments([text]);

        const update: CheckpointRecord = {
          thread_id: config?.configurable?.thread_id,
          checkpoint_id: checkpoint.id,
          parent_id: config?.configurable?.checkpoint_id,
          checkpoint: this.serde.stringify(checkpoint),
          metadata: this.serde.stringify(metadata),
          embedding: embeddings[0] || [],
          history: text,
          timestamp: new Date(),
        };

        await this.collection.insertOne(update);
      }
    } catch (error) {
      console.error("Error saving checkpoint", error);
      throw error;
    }

    return {
      configurable: {
        thread_id: config?.configurable?.thread_id,
        checkpoint_id: checkpoint?.id,
      },
    };
  }

  async closeConnection(): Promise<void> {
    await this.client.close();
  }
}
```

### 5. Conversation Management with LangGraph

First, we need to create tools, we'll use TavilySearch for the internet acsess and our `getHistory` function as long term memory

```typescript
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";

const { TAVILY_API_KEY } = env;

const searchTool = new TavilySearchResults({
  maxResults: 1,
  apiKey: TAVILY_API_KEY,
});

const tools = [getHistory, searchTool];
```

Then define the Graph

```typescript
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { ChatAnthropic } from "@langchain/anthropic";
import { END, START, StateGraph, StateGraphArgs } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";

// Define the state interface
interface AgentState {
  messages: HumanMessage[];
}

// Define the graph state
const graphState: StateGraphArgs<AgentState>["channels"] = {
  messages: {
    value: (x: HumanMessage[], y: HumanMessage[]) => x.concat(y),
    default: () => [
      new SystemMessage(
        `You are helpful assistent. You have memory shared between conversations, stored in database\n
          Please, check current messages state first!\n
          If you still miss it please don't respond, until you tried to use get_history tool. Prompt it for vector searh.\n
          This is start of the new conversation`
      ),
    ],
  },
};

const { ANTHROPIC_API_KEY } = env;

// initialize model
const model = new ChatAnthropic({
  model: "claude-3-sonnet-20240229",
  temperature: 0,
  apiKey: ANTHROPIC_API_KEY,
});

// Bind tools to the model
const toolNode = new ToolNode(tools);
const boundModel = model.bindTools(tools);

// Define the function that determines whether to continue or not
function shouldContinue(state: AgentState): "tools" | typeof END {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;

  // If the LLM makes a tool call, then we route to the "tools" node
  if (lastMessage.tool_calls?.length) {
    return "tools";
  }
  // Otherwise, we stop (reply to the user)
  return END;
}

// Define the function that calls the model
async function callModel(state: AgentState, congig: RunnableConfig) {
  const messages = state.messages;

  const response = await boundModel.invoke(messages, config);

  // We return a list, because this will get added to the existing list
  return { messages: [response] };
}

// Define a new graph
const workflow = new StateGraph<AgentState>({ channels: graphState })
  .addNode("agent", callModel)
  .addNode("tools", toolNode)

  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tools", "agent");
```

### 5. Define user interface

We'll use function that accepts compiled graph and user message,
at the very first call it will put system message to instruct model how to use tools

```typescript
import { HumanMessage } from "@langchain/core/messages";
import { CompiledStateGraph } from "@langchain/langgraph";

async function writeUserMessage(
  app: CompiledStateGraph<AgentState>,
  userMessage: string
) {
  try {
    console.log("\x1b[31m%s\x1b[0m", userMessage);

    const inputs = {
      messages: [new HumanMessage(userMessage)],
    };

    for await (const event of await app.stream(inputs, {
      streamMode: "values",
    })) {
      const lastMessage = event.messages[event.messages.length - 1];
      // console.log("\x1b[32m%s\x1b[0m", 'DEBUG', lastMessage)
      if (lastMessage.tool_calls?.length === 0) {
        // final answer
        console.log("\x1b[36m%s\x1b[0m", lastMessage.content);
      }
    }
  } catch (e) {
    console.log(e);
  }
}
```

### 5. Testing the AI Agent

Start a new conversation

```typescript
import { ObjectId } from "mongodb";

// cleanup history
await collection.deleteMany({});

const checkpointer = MongoSaver.fromConnString();

let config = {
  configurable: {
    thread_id: new ObjectId().valueOf().toString(),
  },
};

const app: CompiledStateGraph<AgentState> = workflow
  .compile({ checkpointer })
  .withConfig(config);
```

Test short-term memory, getting previous message from the graph state

```typescript
await writeUserMessage(app, "Hello. My name is vasily");
```

```text
[DEBUG] history search: vasily
Hello Vasily, it's nice to meet you! Since this is the start of a new conversation, I don't have any prior context about you. Please feel free to ask me anything or let me know if there is a particular topic you'd like to discuss. I'm an AI assistant created by Anthropic to be helpful, honest and harmless.
```

```typescript
await writeUserMessage(app, "do you remember my name");
```

```text
Yes, I remember your name is Vasily.
```

```typescript
await writeUserMessage(app, "what is current weather in ny");
```

```text
According to the latest search results, the current weather in New York City today is sunny with a temperature of around 88°F (31°C). The wind is light from the west-southwest at around 4-6 mph. Overall it appears to be a warm and sunny summer day in NYC.
Let me know if you need any other details about the current weather conditions there!
```

```typescript
await writeUserMessage(app, "Tell me a story");
```

```text
    Here's a little story for you Vasily:

    Once upon a time, there was a young adventurer named Alex who loved exploring the great outdoors. One summer, Alex decided to go on a hiking trip through the mountains.

    After driving for hours down winding roads, Alex finally arrived at the trailhead. With a backpack full of supplies, Alex set off into the wilderness. The trail wound its way up the mountainside, through fields of wildflowers and past babbling brooks.

    As Alex climbed higher, the trees thinned out and the views became more and more breathtaking. Snow-capped peaks rose majestically in the distance. Alex stopped to catch their breath and admire the scenery.

    Suddenly, a rustling came from the bushes nearby. Alex froze, wondering if it was a bear or other wild animal. But then a cute little chipmunk scampered out onto the trail! It stood up on its hind legs, stuffing its cheeks with seeds and nuts before scurrying away.

    Alex laughed at the silly chipmunk and continued on the trail, feeling grateful to be out in nature away from the hustle and bustle of the city. Who knows what other adventures the day would bring?

    I hope you enjoyed that little story, Vasily! I aimed to create a lighthearted narrative about an adventurer exploring the mountains. Let me know if you'd like to hear another story sometime.
```

Now let's start new conversation thread and see if it remembers something!

```typescript
let config = {
  configurable: {
    thread_id: new ObjectId().valueOf().toString(),
  },
};

const app: CompiledStateGraph<AgentState> = workflow
  .compile({ checkpointer })
  .withConfig(config);
```

```typescript
await writeUserMessage(app, "What was the story");
```

```text
[DEBUG] history search: story
  Based on the search results, it seems the story I previously told was about a young adventurer named Alex who went on a hiking trip through the mountains. The story described Alex's journey along the trail, encountering things like wildflowers, streams, mountain views, and a cute chipmunk. It was meant to be a lighthearted, descriptive tale capturing the wonder of being out in nature.

  ince you asked "What was the story", I've provided a summary of the key details from the story I had told earlier in our conversation. Please let me know if you need any clarification or have additional questions!
```

```typescript
await writeUserMessage(
  app,
  "I asked you about weather before, please remind me"
);

````

```text
[DEBUG] history search: weather
  Unfortunately, I could not find any previous messages in our conversation history about you asking me about the weather before this current exchange. The search results only show that you asked "what is current weather in ny", to which I provided the current weather details for New York City.

  If you had asked about weather at an earlier point, those messages do not seem to be present in our conversation history that I have access to. Please let me know if you have any other details that could help me locate a previous weather-related query from you. Otherwise, I do not have enough context to remind you about a prior weather discussion.
```

### 6. Summing up

We implemented Agent with Memory and ability to search long-term history using MongoDB for both.
Check out this links for the next steps

- [langgraph.js](https://langchain-ai.github.io/langgraphjs/)
- [langchain.js](https://js.langchain.com/v0.2/docs/introduction/)
- [Deno jupyter notebook setup](https://docs.deno.com/runtime/manual/tools/jupyter/)
- [MongoDB Tutorials](https://www.mongodb.com/developer/tutorials/)

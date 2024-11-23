import { useState } from 'react'
import './App.css'
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';
import { MainContainer, ChatContainer, MessageList, Message, MessageInput, TypingIndicator } from '@chatscope/chat-ui-kit-react';

const API_KEY = "FrMUXGjo1yvTgwUcd2pNpzv8sO84yvIeRgoKQgwvDQvNKQdYOa0lJQQJ99AKACYeBjFXJ3w3AAABACOG25GJ";
const ENDPOINT = "https://penglaishan.openai.azure.com/";
const DEPLOYMENT_NAME = "gpt-35-turbo";
const API_VERSION = "2024-05-01-preview";

const systemMessage = {
  "role": "system", "content": "You are a Entrepreneurship Education Manager. You are a professional in Retail Store, Business and Supply Chain Management."
}

function App() {
  const [messages, setMessages] = useState([
    {
      message: "Hello, I'm STAT/APAI4011 Entrepreneurship Education Chatbot! Ask me about Retail Store, Business and Supply Chain Management!",
      sentTime: "just now",
      sender: "ChatGPT"
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = async (message) => {
    const newMessage = {
      message,
      direction: 'outgoing',
      sender: "user"
    };

    const newMessages = [...messages, newMessage];
    
    setMessages(newMessages);

    setIsTyping(true);
    await processMessageToChatGPT(newMessages);
  };

  async function processMessageToChatGPT(chatMessages) {
    let apiMessages = chatMessages.map((messageObject) => {
      let role = "";
      if (messageObject.sender === "ChatGPT") {
        role = "assistant";
      } else {
        role = "user";
      }
      return { role: role, content: messageObject.message}
    });

    const apiRequestBody = {
      "messages": [
        systemMessage,
        ...apiMessages
      ],
      "max_tokens": 800,
      "temperature": 0.7,
    }

    await fetch(`${ENDPOINT}openai/deployments/${DEPLOYMENT_NAME}/chat/completions?api-version=${API_VERSION}`, 
    {
      method: "POST",
      headers: {
        "api-key": API_KEY,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(apiRequestBody)
    })
    .then((response) => response.json())
    .then((data) => {
      console.log(data);
      setMessages([...chatMessages, {
        message: data.choices[0].message.content,
        sender: "ChatGPT"
      }]);
      setIsTyping(false);
    })
    .catch((error) => {
      console.error("Error:", error);
      setIsTyping(false);
    });
  }

  return (
    <div className="App" style={{backdropFilter: "blur(10px)"}}>
      <div style={{ height: "80vh", width: "80vh", borderRadius: "30px"}}>
        <MainContainer style={{ borderRadius: "30px", backgroundColor: "#2A403D", border: "none"}}>
          <ChatContainer style={{backgroundColor: "#2A403D"}}>       
            <MessageList
              scrollBehavior="smooth" 
              typingIndicator={isTyping ? <TypingIndicator content="Chatbot is typing" /> : null}
              style={{ backgroundColor: "#2A403D", marginTop: "30px" }}
            >
              {messages.map((message, i) => {
                console.log(message)
                return <Message key={i} model={message} style={{textAlign: "left"}}/>
              })}
            </MessageList>
            <MessageInput placeholder="Type message here" onSend={handleSend} />        
          </ChatContainer>
        </MainContainer>
      </div>
    </div>
  )
}

export default App
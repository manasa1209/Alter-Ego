const express = require('express');
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require('@google/generative-ai');
const dotenv = require('dotenv').config();

const app = express();
const port = process.env.PORT || 3001;
app.use(express.json());
const MODEL_NAME = "gemini-pro";
const API_KEY = process.env.API_KEY;

async function runChat(userInput) {
  const genAI = new GoogleGenerativeAI(API_KEY);
  const model = genAI.getGenerativeModel({ model: MODEL_NAME });

  const generationConfig = {
    temperature: 0.7,
    topK: 1,
    topP: 1,
    maxOutputTokens: 150,  // Reduce the max tokens for shorter responses
  };

  const safetySettings = [
    {
      category: HarmCategory.HARM_CATEGORY_HARASSMENT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
      category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
      category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
    {
      category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
  ];

  const chat = model.startChat({
    generationConfig,
    safetySettings,
    history: [
      {
        role: "user",
        parts: [{ text: "Hey" }],
      },
      {
        role: "model",
        parts: [{ text: "Hey there! 😊 I'm Aptha Mitra, your friendly AI buddy. What's your name?" }],
      },
      {
        role: "user",
        parts: [{ text: "Hi" }],
      },
      {
        role: "model",
        parts: [{ text: "Hiya! I'm here to listen and help. What's up?" }],
      },
      {
        role: "user",
        parts: [{ text: "I had a bad day today. I was chased by a dog while going to school and was scolded by my teacher as didn't finish my assignments" }],
      },
      {
        role: "model",
        parts: [{ text: "Oh no! That sounds like a really tough day. It's understandable that you're feeling down. Here's a virtual hug for you: 🤗 Did you get any injuries? If possible, try to report the dog to the local authorities to prevent it from harming others. And I know it's frustrating when you get scolded, especially for something you couldn't control. Remember that your teacher is human too and may have had a bad day as well. Try to talk to your teacher calmly and P.S. Next time you see a dog, try waving a white flag. Just kidding! Stay safe out there!" }],
      },
      {
        role: "user",
        parts: [{ text: "how do i focus on my studies? Every time i sit to study, i want to do anything other than studying." }],
      },
      {
        role: "model",
        parts: [{ text: "Distractions in today's digital world can easily compete with the demands of studying. Set Clear Goals and Priorities, create a Conducive Study Environment by choosing a quiet and distraction-free space, free from noise and clutter, and break down Large Tasks into smaller, manageable chunks. This makes the task seem less daunting. Remember, you're not alone in this, and hey, even superheroes need to study sometimes haha!" }],
      },
      {
        role: "user",
        parts: [{ text: "It was an amazing day todayyyy! Today was so much fun." }],
      },
      {
        role: "model",
        parts: [{ text: "Yay! I'm so happy to hear that! 😊 What made today so amazing? Did something wonderful happen? Share your joy with me! I'm all ears (or should I say, 'all circuits'?) haha! P.S. Your enthusiasm is contagious! Keep shining brightly! 🌟" }],
      },
      {
        role: "user",
        parts: [{ text: "I ate something today and that was the best food I have ever eaten! " }],
      },
      {
        role: "model",
        parts: [{ text: "Oh, that sounds amazing! I'm always on the lookout for great food recommendations. What did you have the pleasure of indulging in? Was it a dish you've tried before or something completely new? Share your culinary adventure with me! 😋" }],
      },
    ],
  });

  try {
    const result = await chat.sendMessage(userInput);
    let response = result.response.text();

    // Remove markdown formatting and ensure each point starts on a new line
    response = response
      .replace(/\*\*/g, '')  // Remove ** for bold
      .replace(/\*/g, '')    // Remove * for italics
      .replace(/(\d+\.\s)/g, '\n$1')  // Ensure points start on a new line
      .replace(/\n+/g, '\n'); // Remove extra newlines

    
    return response;
  } catch (error) {
    if (error.response && error.response.candidates && error.response.candidates[0].blockReason) {
      console.warn('Blocked response:', error.response.candidates[0].blockReason);
      // Attempt to regenerate the response with different settings if blocked
      return "I'm sorry, but I can't respond to that. Could you please ask something else?";
    } else {
      throw error;
    }
  }
}

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/index.html');
});

app.get('/symbol_for_loading.gif', (req, res) => {
  res.sendFile(__dirname + '/symbol_for_loading.gif');
});

app.post('/chat', async (req, res) => {
  try {
    const userInput = req.body?.userInput;
    console.log('incoming /chat req', userInput);
    if (!userInput) {
      return res.status(400).json({ error: 'Invalid request body' });
    }

    const response = await runChat(userInput);
    res.json({ response });
  } catch (error) {
    console.error('Error in chat endpoint:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});

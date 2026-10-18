import 'dotenv/config';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';
import {
  buildInstantOfficialDocument,
  generateLocalSmartTutorReply,
  getDefaultFlashcardsForSubject,
  getSyllabusTopicsForClassSubject,
  LearnedMemoryItem,
  OfficialDocType,
  SiteResourceContext,
  TeacherMetadata,
} from './src/data/curriculumData.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(express.json({ limit: '15mb' }));

function getGenAIClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY' || apiKey.trim().length < 8) {
    return null;
  }
  return new GoogleGenAI({
    apiKey: apiKey.trim(),
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Proxy endpoint for Google Sheets CSV so client can sync reliably
app.get('/api/sheets/:sheetId', async (req, res) => {
  const { sheetId } = req.params;
  const endpoints = [
    `https://docs.google.com/spreadsheets/d/${encodeURIComponent(sheetId)}/gviz/tq?tqx=out:csv`,
    `https://docs.google.com/spreadsheets/d/${encodeURIComponent(sheetId)}/export?format=csv`,
  ];

  for (const url of endpoints) {
    try {
      const response = await fetch(url, {
        headers: { 'User-Agent': 'Mozilla/5.0' },
      });
      if (response.ok) {
        const text = await response.text();
        if (text && !text.includes('<!DOCTYPE html>') && text.length > 20) {
          res.setHeader('Content-Type', 'text/csv; charset=utf-8');
          return res.send(text);
        }
      }
    } catch {
      // try next endpoint
    }
  }
  res.status(502).json({ error: 'Unable to fetch Google Sheet CSV' });
});

// Moha AI Chat & Multimodal Question Solver Endpoint
app.post('/api/moha/chat', async (req, res) => {
  const {
    prompt,
    cls = 'Form 3',
    subject = 'Basic Mathematics',
    image,
    history = [],
    learnedMemory = [],
    siteResources,
  } = req.body as {
    prompt: string;
    cls?: string;
    subject?: string;
    image?: { mimeType: string; base64Data: string } | null;
    history?: Array<{ id?: string; role: 'user' | 'ai'; text: string }>;
    learnedMemory?: LearnedMemoryItem[];
    siteResources?: SiteResourceContext;
  };

  const userPrompt = (prompt || 'Solve this question step-by-step and show the NECTA Marking Scheme.').trim();

  // Check if user is explicitly querying a learned item first
  const lower = userPrompt.toLowerCase();
  for (const item of learnedMemory) {
    if (lower === item.topic.toLowerCase()) {
      return res.json({
        reply: `🧠 **Moha AI (Learned Memory — ${item.topic}):**\n\n${item.content}\n\n*(Saved on ${item.timestamp})*`,
        source: 'memory',
      });
    }
  }

  const ai = getGenAIClient();
  const syllabusTopics = getSyllabusTopicsForClassSubject(cls, subject);
  const learnedContext =
    learnedMemory.length > 0
      ? `\n\nTeacher/Student Custom Learned Memory:\n${learnedMemory
          .slice(0, 20)
          .map((m) => `- ${m.topic} (${m.cls} ${m.subject}): ${m.content}`)
          .join('\n')}`
      : '';

  const siteResourcesContext = siteResources
    ? `\n\nUploaded Elimika Website Resources Integrated for ${cls} ${subject} & Related Subjects:
- TIE Textbooks Available: ${
        siteResources.books.map((b) => `${b.title} (${b.form} ${b.subject})`).join('; ') || 'Standard TIE Textbooks'
      }
- Official Syllabuses Available: ${
        siteResources.syllabuses.map((s) => `${s.title} (${s.form} ${s.subject})`).join('; ') || 'Official TIE Syllabus'
      }
- Past Examination Papers Available: ${
        siteResources.papers.map((p) => `${p.title} [${p.category} ${p.year}]`).join('; ') || 'NECTA Past Papers'
      }`
    : '';

  if (ai) {
    try {
      const systemInstruction = `You are Moha AI, the official Tanzanian TIE (Tanzania Institute of Education) Competence-Based Curriculum & NECTA (National Examinations Council of Tanzania) Examiner Assistant for ${cls} ${subject}.
Official TIE Syllabus Competences for ${cls} ${subject}:
${syllabusTopics.map((t) => `- ${t}`).join('\n')}${siteResourcesContext}${learnedContext}

Formatting & Pedagogical Rules:
1. Give thorough, crystal-clear, step-by-step explanations, lesson notes, lesson plans, or schemes of work tailored to Tanzanian ${cls} ${subject} students and teachers.
2. Actively integrate and reference the uploaded TIE Textbooks, Official Syllabuses, and Past Examination Papers available on the Elimika website.
3. CLEAR TABLES: Whenever the user asks for a table, comparison, experimental results (e.g., Physics Ohm's Law, Chemistry Titration Burette Table), Mathematics table of values/statistics, or Book-Keeping/Accounts ledgers, ALWAYS draw a clean, well-aligned Markdown table using pipe syntax (| Column 1 | Column 2 |).
4. When solving mathematical, physics, or chemistry problems, write clean LaTeX equations enclosed in $...$ for inline math and $$...$$ for block equations.
5. Include NECTA Marking Scheme mark allocations (e.g., [02 Marks], [01 Mark]) for steps and key points.
6. Use **bold** headings and bullet points (•) for readability.`;

      const recentHistoryParts = history
        .filter((h) => !h.text.includes('official-tie-doc') && h.id !== 'msg-welcome')
        .slice(-6)
        .map((h) => `${h.role === 'user' ? 'Student/Teacher' : 'Moha AI'}: ${h.text}`)
        .join('\n\n');

      const fullPromptText = recentHistoryParts
        ? `Previous conversation context:\n${recentHistoryParts}\n\nCurrent Request (${cls} ${subject}):\n${userPrompt}`
        : userPrompt;

      const parts: Array<{ text?: string; inlineData?: { mimeType: string; data: string } }> = [];
      if (image && image.base64Data && image.mimeType) {
        parts.push({
          inlineData: {
            mimeType: image.mimeType,
            data: image.base64Data,
          },
        });
      }
      parts.push({ text: fullPromptText });

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: { parts },
        config: {
          systemInstruction,
          temperature: 0.4,
        },
      });

      const replyText = response.text;
      if (replyText && replyText.trim().length > 0) {
        return res.json({ reply: replyText.trim(), source: 'gemini' });
      }
    } catch (err) {
      console.error('Gemini chat error, falling back to local curriculum engine:', err);
    }
  }

  // Fallback to built-in curriculum engine
  const fallbackReply = generateLocalSmartTutorReply(userPrompt, cls, subject, learnedMemory, siteResources);
  return res.json({ reply: fallbackReply, source: 'local' });
});

// Moha AI Official TIE / PO-RALG / NECTA Document & Lesson Notes Generator Endpoint
app.post('/api/moha/generate-doc', async (req, res) => {
  const { type, meta } = req.body as {
    type: OfficialDocType;
    meta: TeacherMetadata;
  };

  const resourceSummary = meta.siteResources
    ? `Uploaded Website Books: ${meta.siteResources.books.map((b) => b.title).join(', ')}. Uploaded Syllabuses: ${meta.siteResources.syllabuses.map((s) => s.title).join(', ')}. Uploaded Past Papers: ${meta.siteResources.papers.map((p) => `${p.title} (${p.year})`).join(', ')}.`
    : '';

  const ai = getGenAIClient();
  if (ai && (type === 'lessonplan' || type === 'markingscheme' || type === 'lessonnotes')) {
    try {
      if (type === 'lessonplan') {
        const response = await ai.models.generateContent({
          model: 'gemini-3.8-flash',
          contents: `Create specific, curriculum-authentic Tanzanian TIE 4-stage lesson plan activities for Class: ${meta.cls}, Subject: ${meta.subject}, Main Topic: ${meta.topic}, Sub-Topic: ${meta.subTopic}, Present Students: ${meta.presTotal}. Integrate these website resources: ${resourceSummary}`,
          config: {
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                mainCompetence: { type: Type.STRING },
                specificCompetence: { type: Type.STRING },
                introTeacher: { type: Type.STRING },
                introStudent: { type: Type.STRING },
                compTeacher: { type: Type.STRING },
                compStudent: { type: Type.STRING },
                designTeacher: { type: Type.STRING },
                designStudent: { type: Type.STRING },
                realTeacher: { type: Type.STRING },
                realStudent: { type: Type.STRING },
                remarks: { type: Type.STRING },
              },
              required: [
                'mainCompetence',
                'specificCompetence',
                'introTeacher',
                'introStudent',
                'compTeacher',
                'compStudent',
                'designTeacher',
                'designStudent',
                'realTeacher',
                'realStudent',
                'remarks',
              ],
            },
          },
        });
        const jsonText = response.text?.trim();
        if (jsonText) {
          const parsed = JSON.parse(jsonText);
          const docHtml = buildInstantOfficialDocument('lessonplan', meta, parsed);
          return res.json({ html: docHtml, source: 'gemini' });
        }
      } else if (type === 'lessonnotes') {
        const response = await ai.models.generateContent({
          model: 'gemini-3.8-flash',
          contents: `Create comprehensive official Tanzanian TIE Lesson Notes for Class: ${meta.cls}, Subject: ${meta.subject}, Main Topic: ${meta.topic}, Sub-Topic: ${meta.subTopic}. Integrate concepts from the TIE textbook, official syllabus, and past examination papers (${resourceSummary}). Include 3 detailed keyConcepts, 2 step-by-step workedExamples, and 3 NECTA-style practiceQuestions with marking hints.`,
          config: {
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                keyConcepts: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      heading: { type: Type.STRING },
                      explanation: { type: Type.STRING },
                    },
                    required: ['heading', 'explanation'],
                  },
                },
                workedExamples: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      problem: { type: Type.STRING },
                      solution: { type: Type.STRING },
                    },
                    required: ['problem', 'solution'],
                  },
                },
                practiceQuestions: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      question: { type: Type.STRING },
                      answerHint: { type: Type.STRING },
                    },
                    required: ['question', 'answerHint'],
                  },
                },
              },
              required: ['keyConcepts', 'workedExamples', 'practiceQuestions'],
            },
          },
        });
        const jsonText = response.text?.trim();
        if (jsonText) {
          const parsed = JSON.parse(jsonText);
          const docHtml = buildInstantOfficialDocument('lessonnotes', meta, parsed);
          return res.json({ html: docHtml, source: 'gemini' });
        }
      } else if (type === 'markingscheme') {
        const response = await ai.models.generateContent({
          model: 'gemini-3.8-flash',
          contents: `Create 4 detailed NECTA examination questions and marking rubric items for Class: ${meta.cls}, Subject: ${meta.subject}, Main Topic: ${meta.topic}, Sub-Topic: ${meta.subTopic}. Reference uploaded past papers: ${resourceSummary}. Use <br/> between rubric bullet points.`,
          config: {
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                markingItems: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      qn: { type: Type.STRING },
                      item: { type: Type.STRING },
                      rubric: { type: Type.STRING },
                      marks: { type: Type.STRING },
                    },
                    required: ['qn', 'item', 'rubric', 'marks'],
                  },
                },
              },
              required: ['markingItems'],
            },
          },
        });
        const jsonText = response.text?.trim();
        if (jsonText) {
          const parsed = JSON.parse(jsonText);
          const docHtml = buildInstantOfficialDocument('markingscheme', meta, parsed);
          return res.json({ html: docHtml, source: 'gemini' });
        }
      }
    } catch (err) {
      console.error('Gemini doc enhancement fallback:', err);
    }
  }

  const html = buildInstantOfficialDocument(type, meta);
  return res.json({ html, source: 'local' });
});

// Moha AI Interactive Quiz Generator Endpoint
app.post('/api/moha/quiz', async (req, res) => {
  const { cls = 'Form 3', subject = 'Basic Mathematics' } = req.body as {
    cls?: string;
    subject?: string;
  };

  const syllabusTopics = getSyllabusTopicsForClassSubject(cls, subject);
  const ai = getGenAIClient();

  if (ai) {
    try {
      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: `Generate 4 high-quality NECTA multiple-choice practice questions for Tanzanian ${cls} ${subject} covering these official TIE syllabus topics: ${syllabusTopics.join(
          '; '
        )}. Each question must have 4 options, correctIndex (0 to 3), a clear explanation, and the specific syllabus topic name it tests.`,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                question: { type: Type.STRING },
                options: {
                  type: Type.ARRAY,
                  items: { type: Type.STRING },
                },
                correctIndex: { type: Type.INTEGER },
                explanation: { type: Type.STRING },
                topic: { type: Type.STRING },
              },
              required: ['question', 'options', 'correctIndex', 'explanation', 'topic'],
            },
          },
        },
      });
      const jsonText = response.text?.trim();
      if (jsonText) {
        const questions = JSON.parse(jsonText);
        if (Array.isArray(questions) && questions.length > 0) {
          return res.json({ questions, source: 'gemini' });
        }
      }
    } catch (err) {
      console.error('Gemini quiz fallback:', err);
    }
  }

  const fallbackQuestions = [
    {
      question: `Which curriculum framework is officially implemented by TIE for ${cls} ${subject} in Tanzania?`,
      options: [
        'Competence-Based Curriculum (CBC)',
        'Content-Only Memorization System',
        'Non-Structured Elective System',
        'Ungraded Assessment System',
      ],
      correctIndex: 0,
      explanation:
        'The Tanzania Institute of Education (TIE) implements the Competence-Based Curriculum (CBC) focusing on practical student competences.',
      topic: syllabusTopics[0] || `1.0 Core Principles of ${subject}`,
    },
    {
      question: `In the official 4-stage TIE Lesson Plan for ${subject}, which stage activates students' prior knowledge?`,
      options: ['Realisation', 'Design', 'Introduction', 'Competence Development'],
      correctIndex: 2,
      explanation:
        'Stage 1 (Introduction) connects students’ prior knowledge and experiences to the new competence.',
      topic: syllabusTopics[1] || `2.0 Practical Applications in ${subject}`,
    },
    {
      question: `Which of the following is a core TIE syllabus competence area in ${cls} ${subject}?`,
      options: [
        syllabusTopics[0] || `1.0 Core Principles of ${subject}`,
        'Unrelated Foreign History',
        'Non-Scientific Speculation',
        'Optional Extracurricular Trivia',
      ],
      correctIndex: 0,
      explanation: `${
        syllabusTopics[0] || `Core Principles of ${subject}`
      } is an official competence in the TIE syllabus for ${cls} ${subject}.`,
      topic: syllabusTopics[0] || `1.0 Core Principles of ${subject}`,
    },
    {
      question: `When answering structured NECTA examination problems in ${subject}, how are marks primarily awarded?`,
      options: [
        'Only for the final number without working',
        'Step-by-step method marks (formula, substitution, and final answer with SI units)',
        'Length of handwriting only',
        'Guessing without showing procedures',
      ],
      correctIndex: 1,
      explanation:
        'NECTA marking schemes award marks for extracting given data, stating the correct formula/principle, step-by-step substitution, and the final answer with units.',
      topic: syllabusTopics[2] || `3.0 Problem Solving in ${subject}`,
    },
  ];

  return res.json({ questions: fallbackQuestions, source: 'local' });
});

// Moha AI Revision Flashcards Generator Endpoint
app.post('/api/moha/flashcards', async (req, res) => {
  const { cls = 'Form 3', subject = 'Basic Mathematics' } = req.body as {
    cls?: string;
    subject?: string;
  };

  const syllabusTopics = getSyllabusTopicsForClassSubject(cls, subject);
  const ai = getGenAIClient();

  if (ai) {
    try {
      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: `Generate 5 high-yield revision flashcards for Tanzanian ${cls} ${subject} covering these official TIE syllabus topics: ${syllabusTopics.join(
          '; '
        )}. Each flashcard must have an id, topic, front (clear question or concept prompt), back (concise TIE/NECTA model answer), and formulaOrKeyPoint.`,
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING },
                topic: { type: Type.STRING },
                front: { type: Type.STRING },
                back: { type: Type.STRING },
                formulaOrKeyPoint: { type: Type.STRING },
              },
              required: ['id', 'topic', 'front', 'back', 'formulaOrKeyPoint'],
            },
          },
        },
      });
      const jsonText = response.text?.trim();
      if (jsonText) {
        const cards = JSON.parse(jsonText);
        if (Array.isArray(cards) && cards.length > 0) {
          return res.json({ flashcards: cards, source: 'gemini' });
        }
      }
    } catch (err) {
      console.error('Gemini flashcards fallback:', err);
    }
  }

  return res.json({
    flashcards: getDefaultFlashcardsForSubject(cls, subject),
    source: 'local',
  });
});

async function startServer() {
  const PORT = 3000;

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Elimika Tanzania server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();

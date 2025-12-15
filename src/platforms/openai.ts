import { generateText, streamText, generateObject } from 'ai';
import { createStreamableValue } from '@ai-sdk/rsc';
import { createOpenAI } from '@ai-sdk/openai';
import { createDeepSeek } from '@ai-sdk/deepseek';
import { OPENAI_BASE_URL, OPENAI_SECRET_KEY, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, AI_SECRET_KEY, AI_BASE_URL } from '@/app/config';
import { removeBase64Prefix } from '@/utility/image';
import { cleanUpAiTextResponse } from '@/photo/ai';
import {
  checkRateLimitAndThrow as _checkRateLimitAndThrow,
} from '@/platforms/rate-limit';
import { z } from 'zod';

// DeepSeek configuration
export const IS_DEEPSEEK = DEEPSEEK_API_KEY && (DEEPSEEK_BASE_URL?.includes('deepseek') || OPENAI_BASE_URL?.includes('deepseek'));

const checkRateLimitAndThrow = (isBatch?: boolean) =>
  _checkRateLimitAndThrow({
    identifier: 'openai-image-query',
    ...isBatch && { tokens: 1200, duration: '1d' },
  });

// Model configuration
const OPENAI_MODEL = 'gpt-5.1';
const DEEPSEEK_MODEL = 'deepseek-chat';

// Create AI client instances
const openai = OPENAI_SECRET_KEY
  ? createOpenAI({
      apiKey: OPENAI_SECRET_KEY,
      ...OPENAI_BASE_URL && { baseURL: OPENAI_BASE_URL },
    })
  : undefined;

const deepseek = AI_SECRET_KEY
  ? createDeepSeek({
      apiKey: AI_SECRET_KEY,
      ...AI_BASE_URL && { baseURL: AI_BASE_URL },
    })
  : undefined;

// Use DeepSeek if configured, otherwise fall back to OpenAI
const aiClient = IS_DEEPSEEK ? deepseek : openai;
const MODEL = IS_DEEPSEEK ? DEEPSEEK_MODEL : OPENAI_MODEL;

const getImageTextArgs = (
  imageBase64: string,
  query: string,
): (
  Parameters<typeof streamText>[0] &
  Parameters<typeof generateText>[0]
) | undefined => aiClient ? {
  model: aiClient(MODEL),
  messages: [{
    'role': 'user',
    'content': [
      {
        'type': 'text',
        'text': query,
      }, {
        'type': 'image',
        'image': removeBase64Prefix(imageBase64),
      },
    ],
  }],
} : undefined;

export const streamOpenAiImageQuery = async (
  imageBase64: string,
  query: string,
) => {
  await checkRateLimitAndThrow();

  const stream = createStreamableValue('');

  const args = getImageTextArgs(imageBase64, query);

  if (args) {
    (async () => {
      const { textStream } = streamText(args);
      for await (const delta of textStream) {
        stream.update(cleanUpAiTextResponse(delta));
      }
      stream.done();
    })();
  }

  return stream.value;
};

export const generateOpenAiImageQuery = async (
  imageBase64: string,
  query: string,
  isBatch?: boolean,
) => {
  await checkRateLimitAndThrow(isBatch);

  const args = getImageTextArgs(imageBase64, query);

  if (args) {
    return generateText(args)
      .then(({ text }) => cleanUpAiTextResponse(text));
  }
};

export const generateOpenAiImageObjectQuery = async <T extends z.ZodSchema>(
  imageBase64: string,
  query: string,
  schema: T,
  isBatch?: boolean,
): Promise<z.infer<T>> => {
  await checkRateLimitAndThrow(isBatch);

  if (aiClient) {
    return generateObject({
      model: aiClient(MODEL),
      messages: [{
        'role': 'user',
        'content': [
          {
            'type': 'text',
            'text': query,
          }, {
            'type': 'image',
            'image': removeBase64Prefix(imageBase64),
          },
        ],
      }],
      schema,
    }).then(result => Object.fromEntries(Object
      .entries(result.object || {})
      .map(([k, v]) => [k, cleanUpAiTextResponse(v as string)]),
    ) as z.infer<T>);
  } else {
    throw new Error('No AI client available');
  }
};

export const testOpenAiConnection = async () => {
  await checkRateLimitAndThrow();

  if (aiClient) {
    return generateText({
      model: aiClient(MODEL),
      messages: [{
        'role': 'user',
        'content': [
          {
            'type': 'text',
            'text': 'Test connection',
          },
        ],
      }],
    });
  }
};

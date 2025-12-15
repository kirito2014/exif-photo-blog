// Import only what we need to avoid confusion
import { generateText, streamText, generateObject } from 'ai';
import { createStreamableValue } from '@ai-sdk/rsc';
import { createOpenAI } from '@ai-sdk/openai';
import { createDeepSeek } from '@ai-sdk/deepseek';
import { OPENAI_BASE_URL, OPENAI_SECRET_KEY, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL } from '@/app/config';
import { removeBase64Prefix } from '@/utility/image';
import { cleanUpAiTextResponse } from '@/photo/ai';
import {
  checkRateLimitAndThrow as _checkRateLimitAndThrow,
} from '@/platforms/rate-limit';
import { z } from 'zod';

// Validate and normalize base URLs
const normalizeBaseUrl = (url?: string, provider?: 'deepseek' | 'openai'): string | undefined => {
  if (!url) return undefined;
  
  try {
    // Ensure URL has http/https protocol
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
      console.warn(`Invalid ${provider} base URL (missing protocol): ${url}`);
      return undefined;
    }
    
    // Parse to validate format
    const parsed = new URL(url);
    
    // Ensure DeepSeek URL ends with /v1
    if (provider === 'deepseek' && !parsed.pathname.endsWith('/v1')) {
      parsed.pathname = parsed.pathname.endsWith('/') ? parsed.pathname + 'v1' : parsed.pathname + '/v1';
      return parsed.toString();
    }
    
    return url;
  } catch (e) {
    console.warn(`Invalid ${provider} base URL (parse error): ${url}`, e);
    return undefined;
  }
};

const normalizedDeepseekBaseUrl = normalizeBaseUrl(DEEPSEEK_BASE_URL, 'deepseek');
const normalizedOpenaiBaseUrl = normalizeBaseUrl(OPENAI_BASE_URL, 'openai');

// DeepSeek configuration - only use DeepSeek if both API key and valid normalized URL are present
export const IS_DEEPSEEK = DEEPSEEK_API_KEY && !!normalizedDeepseekBaseUrl;

const checkRateLimitAndThrow = (isBatch?: boolean) =>
  _checkRateLimitAndThrow({
    identifier: 'openai-image-query',
    ...isBatch && { tokens: 1200, duration: '1d' },
  });

// Model configuration
const OPENAI_MODEL = 'gpt-5.1';
const DEEPSEEK_MODEL = 'deepseek-chat';

// Create AI client instances
// Only use OPENAI_BASE_URL if it doesn't contain 'deepseek' - prevent cross-provider URL contamination
const safeOpenaiBaseUrl = normalizedOpenaiBaseUrl && !normalizedOpenaiBaseUrl.includes('deepseek') ? normalizedOpenaiBaseUrl : undefined;

const openai = OPENAI_SECRET_KEY
  ? createOpenAI({
      apiKey: OPENAI_SECRET_KEY,
      ...safeOpenaiBaseUrl && { baseURL: safeOpenaiBaseUrl },
    })
  : undefined;

const deepseek = DEEPSEEK_API_KEY && normalizedDeepseekBaseUrl
  ? createDeepSeek({
      apiKey: DEEPSEEK_API_KEY,
      baseURL: normalizedDeepseekBaseUrl,
    })
  : undefined;

// Use DeepSeek if configured, otherwise fall back to OpenAI
const aiClient = IS_DEEPSEEK ? deepseek : openai;
const MODEL = IS_DEEPSEEK ? DEEPSEEK_MODEL : OPENAI_MODEL;

// Add debug logging for troubleshooting
if (process.env.NODE_ENV === 'development' || process.env.ADMIN_DEBUG_TOOLS_ENABLED || process.env.VERCEL_ENV) {
  console.log('AI Client Configuration:', {
    IS_DEEPSEEK,
    hasOpenai: !!openai,
    hasDeepseek: !!deepseek,
    hasAIClient: !!aiClient,
    DEEPSEEK_API_KEY: DEEPSEEK_API_KEY ? 'configured' : 'not configured',
    DEEPSEEK_BASE_URL: DEEPSEEK_BASE_URL,
    normalizedDeepseekBaseUrl: normalizedDeepseekBaseUrl,
    OPENAI_API_KEY: OPENAI_SECRET_KEY ? 'configured' : 'not configured',
    OPENAI_BASE_URL: OPENAI_BASE_URL,
    normalizedOpenaiBaseUrl: normalizedOpenaiBaseUrl,
    safeOpenaiBaseUrl: safeOpenaiBaseUrl,
    MODEL,
    NODE_ENV: process.env.NODE_ENV,
    VERCEL_ENV: process.env.VERCEL_ENV
  });
}

// Debug client creation parameters
if (process.env.VERCEL_ENV) {
  console.log('OpenAI Client Creation Params:', {
    apiKey: OPENAI_SECRET_KEY ? 'present' : 'missing',
    normalizedOpenaiBaseUrl: normalizedOpenaiBaseUrl,
    safeOpenaiBaseUrl: safeOpenaiBaseUrl
  });
  
  console.log('DeepSeek Client Creation Params:', {
    apiKey: DEEPSEEK_API_KEY ? 'present' : 'missing',
    deepseekBaseUrl: DEEPSEEK_BASE_URL,
    normalizedDeepseekBaseUrl: normalizedDeepseekBaseUrl
  });
}

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

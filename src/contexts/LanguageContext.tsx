/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { locales } from '@/i18n';

type Messages = Record<string, any>;
type LanguageContextType = {
  language: string;
  setLanguage: (lang: string) => void;
  messages: Messages;
  supportedLanguages: Record<string, string>;
};

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<string>('en');
  const [messages, setMessages] = useState<Messages>({});
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [supportedLanguages, setSupportedLanguages] = useState({})
  const [defaultLanguage, setDefaultLanguage] = useState('en')

  const detectBrowserLanguage = (): string => {
    try {
      if (typeof window === 'undefined' || typeof navigator === 'undefined') {
        return 'en';
      }

      const browserLang = navigator.language || (navigator as any).userLanguage || '';
      console.log('Detected browser language:', browserLang);

      if (!browserLang) {
        return 'en';
      }

      const langCode = browserLang.split('-')[0].toLowerCase();
      console.log('Extracted language code:', langCode);

      if (locales.includes(langCode as any)) {
        console.log('Language supported, using:', langCode);
        return langCode;
      }

      if (langCode === 'zh') {
        console.log('Chinese language detected');
        if (browserLang.includes('TW') || browserLang.includes('HK')) {
          console.log('Traditional Chinese variant detected');
          return 'zh';
        }
        return 'zh';
      }

      console.log('Language not supported, defaulting to English');
      return 'en';
    } catch (error) {
      console.error('Error detecting browser language:', error);
      return 'en';
    }
  };

  useEffect(() => {
    const getSupportedLanguages = async () => {
      try {
        const response = await fetch('/api/lang/config');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setSupportedLanguages(data.supported_languages);
        setDefaultLanguage(data.default);
      } catch (err) {
        console.error("Failed to fetch auth status:", err);
        const defaultSupportedLanguages = {
          "en": "English",
          "ja": "Japanese (日本語)",
          "zh": "Mandarin Chinese (中文)",
          "zh-tw": "Traditional Chinese (繁體中文)",
          "es": "Spanish (Español)",
          "kr": "Korean (한국어)",
          "vi": "Vietnamese (Tiếng Việt)",
          "pt-br": "Brazilian Portuguese (Português Brasileiro)",
          "fr": "Français (French)",
          "ru": "Русский (Russian)"
        };
        setSupportedLanguages(defaultSupportedLanguages);
        setDefaultLanguage("en");
      }
    }
    getSupportedLanguages();
  }, []);

  useEffect(() => {
    if (Object.keys(supportedLanguages).length > 0) {
      const loadLanguage = async () => {
        try {
          let storedLanguage;
          if (typeof window !== 'undefined') {
            storedLanguage = localStorage.getItem('language');
    
            if (!storedLanguage) {
              console.log('No language in localStorage, detecting browser language');
              storedLanguage = detectBrowserLanguage();
    
              localStorage.setItem('language', storedLanguage);
            }
          } else {
            console.log('Running on server-side, using default language');
            storedLanguage = 'en';
          }
    
          console.log('Supported languages loaded, validating language:', storedLanguage);
          const validLanguage = Object.keys(supportedLanguages).includes(storedLanguage as any) ? storedLanguage : defaultLanguage;
          console.log('Valid language determined:', validLanguage);
    
          const langMessages = (await import(`../messages/${validLanguage}.json`)).default;
    
          setLanguageState(validLanguage);
          setMessages(langMessages);
    
          if (typeof document !== 'undefined') {
            document.documentElement.lang = validLanguage;
          }
        } catch (error) {
          console.error('Failed to load language:', error);
          console.log('Falling back to English due to error');
          const enMessages = (await import('../messages/en.json')).default;
          setMessages(enMessages);
        } finally {
          setIsLoading(false);
        }
      };
      
      loadLanguage();
    }
  }, [supportedLanguages, defaultLanguage]);

  const setLanguage = async (lang: string) => {
    try {
      console.log('Setting language to:', lang);
      const validLanguage = Object.keys(supportedLanguages).includes(lang as any) ? lang : defaultLanguage;

      const langMessages = (await import(`../messages/${validLanguage}.json`)).default;

      setLanguageState(validLanguage);
      setMessages(langMessages);

      if (typeof window !== 'undefined') {
        localStorage.setItem('language', validLanguage);
      }

      if (typeof document !== 'undefined') {
        document.documentElement.lang = validLanguage;
      }
    } catch (error) {
      console.error('Failed to set language:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-100 dark:bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <LanguageContext.Provider value={{ language, setLanguage, messages, supportedLanguages }}>
      {children}
    </LanguageContext.Provider>
  );
}

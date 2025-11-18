import { getRequestConfig } from 'next-intl/server';

export const locales = ['en', 'ja', 'zh', 'es', 'kr', 'vi', 'pt-br'];

export default getRequestConfig(async ({ locale }) => {
  // Use a default locale if the requested one isn't supported
  const safeLocale = locales.includes(locale as string) ? locale : 'en';

  return {
    locale: safeLocale as string,
    messages: (await import(`./messages/${safeLocale}.json`)).default
  };
});

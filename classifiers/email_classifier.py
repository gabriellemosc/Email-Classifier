import os
from openai import OpenAI
from dotenv import load_dotenv      #ENV
import joblib                          #load model
from sklearn.feature_extraction.text import TfidfVectorizer  # vetorization
from sklearn.naive_bayes import MultinomialNB      #  local classifer
from classifiers.preprocess import pre_process_text, extract_keywords


#env: read file and start client connection
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


""" Manual AI agent"""
class EmailClassifier:
    
    def __init__(self):
            #load or train the local model
            if os.path.exists("local_email_model.pkl") and os.path.exists("local_vectorizer.pkl"):
                 self.model = joblib.load("local_email_model.pkl")
                 self.vectorizer = joblib.load("local_vectorizer.pkl")
            else:
                 self.model, self.vectorizer = self.train_local_model()

        
    def train_local_model(self):

            corpus = [
                "Preciso resolver o problema do login", "Erro no sistema ao acessar",
                "Suporte técnico urgente",
                "Reunião de projeto amanhã",
                "Promoção incrível disponível hoje",
                "Ganhe dinheiro fácil agora",
                "A automação do login falhou na etapa 3.",
                "Atualizar a senha do administrador do sistema.",
                "Clique aqui para resgatar seu prêmio imediato!",
                "Verificar os logs de erro da aplicação web.",
                "Você foi selecionado para uma oferta exclusiva de empréstimo.",
                "Documentação técnica da nova funcionalidade.",
                "Alerta: Uso de CPU acima de 90%.",
                "Descubra o segredo dos milionários da noite para o dia.",
                "Solicitação de feature: Relatório de usuários ativos.",
                "Sua fatura está pendente. Pague agora para evitar bloqueio.",
                "Feedback sobre a interface do novo painel."
            ]
        #examples of emails 

            labels = [
                "Produtivo",    
                "Produtivo",    
                "Produtivo",    
                "Produtivo",    
                "Improdutivo",  
                "Improdutivo",  
                "Produtivo",    
                "Produtivo",   
                "Improdutivo",  
                "Produtivo",   
                "Improdutivo", 
                "Produtivo",    
                "Produtivo",    
                "Improdutivo",  
                "Produtivo",  
                "Improdutivo", 
                "Produtivo",    
            ]

            processed = [pre_process_text(c) for c in corpus]
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(processed)
            model = MultinomialNB()
            model.fit(X, labels)
            joblib.dump(model, "local_email_model.pkl")
            joblib.dump(vectorizer, "local_vectorizer.pkl")

            return model, vectorizer

# try to use OPENAI API, else using our fallback 
    def classify_with_openai(self, text):
                cleaned = pre_process_text(text)
                cleaned_lower = cleaned.lower()

                SPAM_KEYWORDS = [
                    "ganhar", "ganhe", "prêmio", "premio", "selecionado", "exclusivo",
                    "clique aqui", "oferta", "promoção", "promocao", "brinde",
                    "sorteado", "grátis", "gratis", "dinheiro", "cupom", "crédito",
                    "credit", "vencedor", "parabéns"
                ]
                if any(word in cleaned_lower for word in SPAM_KEYWORDS):
                    return "Improdutivo"
                

                try:        #use OPENAI
                    response = client.responses.create(
                        model="gpt-4o-mini",
                      input=f"""
                                Classifique o email abaixo como Produtivo ou Improdutivo.
                                Siga estas regras:

                                - "Improdutivo" inclui spam, propaganda, promoções, golpes, sorteios,
                                prêmios, correntes, ofertas suspeitas, assuntos irrelevantes ao trabalho,
                                mensagens genéricas, marketing e emails que não pedem ação real.
                                - "Produtivo" é apenas o que exige ação real, tarefa, reunião, documento,
                                atividade, cliente, chamado, ou comunicação profissional.

                                Não invente nada, responda apenas "Produtivo" ou "Improdutivo".

                                Email:
                                {cleaned}
                                """)

                    result = response.output_text # AI send us back the response

                    if "produtivo" in result.lower():
                        return "Produtivo"
                    else:
                        return "Improdutivo"
                    
                except Exception as e:
                    print("Fail on OPenAI. Using local fallback", e)

                    try:
                          vector = self.vectorizer.transform([cleaned])
                          prediction = self.model.predict(vector)[0]
                          return prediction
                    
                    except Exception as e2:
                          print("Erro no fallback local também", e2)
                          return None
            

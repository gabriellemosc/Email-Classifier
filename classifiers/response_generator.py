import os
from openai import OpenAI
from dotenv import load_dotenv
from classifiers.preprocess import pre_process_text, extract_keywords
from classifiers.email_classifier import EmailClassifier


#env: read file and start client connection
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ResponseGenerator:
    """ Generate a response from each kind of email"""

    def __init__(self):
        self.classifier = EmailClassifier()  

    def generate_response(self, email_text):
        """ Generate a response using OpenAI or local fallback  """

        cleaned = pre_process_text(email_text)              

        classification = self.classifier.classify_with_openai(email_text)  

        if classification is None:                          
            classification = "Produtivo"                    

        # /email is productived  
        if classification == "Produtivo":
            return self._generate_productive_response(cleaned)

        #  email is not 
        else:
      
            return self._generate_unproductive_response(cleaned)
        
    def _generate_productive_response(self, cleaned_text):
        """ Generate a smart response """

        try:    # try OpenAI
            response = client.responses.create(
                model="gpt-4o-mini",                       
                input=f"""
                        Você é um atendente profissional.
                        Gere uma resposta clara, educada e curta para o email abaixo.

                        Email do usuário:
                        {cleaned_text}

                        Responda como se fosse uma empresa real.
                                        """
                                    )

            return response.output_text                     # result

        except Exception as e:
            print("Erro usando OpenAI para resposta produtiva:", e)

            # local fallback  
            return (
                "Olá! Obrigado pelo seu contato. Recebemos sua mensagem e vamos verificar o caso. "
                "Retornaremos em breve com uma solução detalhada."
            )
    
    def _generate_unproductive_response(self, cleaned_text):
        """ Generate response for improductive emails """

        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=f"""
                Gere uma resposta educada e curta para um email que foi classificado como improdutivo
                (ex: propaganda, spam, marketing, golpe). NÃO ofenda o remetente.

                Email do usuário:
                {cleaned_text}

                Responda com cordialidade.
                                """
                            )

            return response.output_text

        except Exception as e:
            print("Erro usando OpenAI para resposta improdutiva:", e)

            # fallback 
            return (
                "Olá! Agradecemos sua mensagem. No momento, não temos interesse na proposta, "
                "mas agradecemos o contato."
            )

if __name__ == "__main__":

    generator = ResponseGenerator()

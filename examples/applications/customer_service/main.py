# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Customer Service Bot - An intelligent customer service application

This application demonstrates a customer service bot that can:
1. Answer frequently asked questions
2. Handle product inquiries
3. Process simple requests
4. Escalate complex issues

Usage:
    python main.py
"""
import sys
import os
from typing import Optional
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agentica import Agent, OpenAIChat


# Define response models for structured output
class CustomerIntent(BaseModel):
    """Classified customer intent."""
    intent: str = Field(..., description="The classified intent: inquiry, complaint, request, feedback, other")
    confidence: float = Field(..., description="Confidence score 0-1")
    summary: str = Field(..., description="Brief summary of the customer's message")


class ServiceResponse(BaseModel):
    """Structured service response."""
    answer: str = Field(..., description="The response to the customer")
    requires_escalation: bool = Field(default=False, description="Whether this needs human escalation")
    escalation_reason: Optional[str] = Field(default=None, description="Reason for escalation if needed")
    suggested_actions: list[str] = Field(default_factory=list, description="Suggested follow-up actions")


# Knowledge base for the customer service bot
KNOWLEDGE_BASE = """
# Company Information
- Company Name: TechCorp Inc.
- Business Hours: Monday-Friday 9:00-18:00
- Support Email: support@techcorp.com
- Phone: 400-123-4567

# Products
1. TechPro X1 - Premium laptop ($1299)
   - 15.6" 4K display
   - Intel i7 processor
   - 16GB RAM, 512GB SSD
   - 2-year warranty

2. TechPad S - Tablet ($599)
   - 11" Retina display
   - 8GB RAM, 256GB storage
   - 1-year warranty

3. TechWatch 3 - Smartwatch ($299)
   - Heart rate monitoring
   - GPS tracking
   - 7-day battery life
   - 1-year warranty

# Policies
- Return Policy: 30-day money-back guarantee
- Warranty: Free repairs for manufacturing defects
- Shipping: Free shipping on orders over $50
- Payment: Credit card, PayPal, bank transfer

# Common Issues
- Password Reset: Visit account settings or contact support
- Order Tracking: Check email for tracking number
- Refund Process: 5-7 business days after return received
"""


def create_intent_classifier():
    """Create an intent classification agent."""
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="IntentClassifier",
        instructions=[
            "You are a customer intent classifier.",
            "Analyze the customer's message and classify their intent.",
            "Categories: inquiry, complaint, request, feedback, other",
        ],
        response_model=CustomerIntent,
    )


def create_service_agent():
    """Create the main customer service agent."""
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="CustomerServiceBot",
        description="A helpful customer service representative",
        instructions=[
            "You are a friendly and professional customer service representative for TechCorp Inc.",
            "Use the following knowledge base to answer questions:",
            "",
            KNOWLEDGE_BASE,
            "",
            "Guidelines:",
            "1. Be polite, helpful, and empathetic",
            "2. Provide accurate information based on the knowledge base",
            "3. If you don't know something, say so and offer to escalate",
            "4. Keep responses concise but complete",
            "5. Always offer additional help at the end",
        ],
        response_model=ServiceResponse,
    )


def handle_customer_message(
    classifier: Agent,
    service_agent: Agent,
    message: str
) -> ServiceResponse:
    """Handle a customer message.
    
    Args:
        classifier: The intent classifier agent
        service_agent: The customer service agent
        message: The customer's message
        
    Returns:
        The service response
    """
    # Step 1: Classify intent
    intent_response = classifier.run_sync(message)
    intent: CustomerIntent = intent_response.content

    print(f"\n[Intent: {intent.intent} (confidence: {intent.confidence:.2f})]")
    print(f"[Summary: {intent.summary}]")

    # Step 2: Generate response
    prompt = f"""
Customer Message: {message}

Classified Intent: {intent.intent}
Summary: {intent.summary}

Please provide a helpful response to this customer.
"""
    response = service_agent.run_sync(prompt)
    return response.content


def interactive_mode():
    """Run the customer service bot in interactive mode."""
    print("\n" + "="*60)
    print("TechCorp Customer Service")
    print("="*60)
    print("\nHello! Welcome to TechCorp customer service.")
    print("How can I help you today?")
    print("\n(Type 'exit' to end the conversation)")

    classifier = create_intent_classifier()
    service_agent = create_service_agent()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("\nThank you for contacting TechCorp. Have a great day!")
                break

            response = handle_customer_message(classifier, service_agent, user_input)

            print(f"\nBot: {response.answer}")

            if response.requires_escalation:
                print(f"\n[Note: This issue has been flagged for escalation]")
                print(f"[Reason: {response.escalation_reason}]")

            if response.suggested_actions:
                print(f"\n[Suggested actions: {', '.join(response.suggested_actions)}]")

        except KeyboardInterrupt:
            print("\n\nThank you for contacting TechCorp. Have a great day!")
            break


def demo_mode():
    """Run demo with sample customer messages."""
    print("="*60)
    print("Customer Service Bot - Demo Mode")
    print("="*60)

    classifier = create_intent_classifier()
    service_agent = create_service_agent()

    sample_messages = [
        "What are your business hours?",
        "I want to return my TechPro X1, it's been 2 weeks since I bought it.",
        "The screen on my TechWatch 3 is cracked. What can I do?",
        "How much does the TechPad S cost?",
        "I've been waiting 2 weeks for my refund and still haven't received it!",
    ]

    for message in sample_messages:
        print(f"\n{'='*60}")
        print(f"Customer: {message}")
        print("-"*60)

        response = handle_customer_message(classifier, service_agent, message)

        print(f"\nBot: {response.answer}")

        if response.requires_escalation:
            print(f"\n[Escalation needed: {response.escalation_reason}]")


def main():
    """Main entry point."""
    # Run demo mode
    demo_mode()

    # Uncomment to run interactive mode
    # interactive_mode()


if __name__ == "__main__":
    main()

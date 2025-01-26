import asyncio
import aiohttp
import time
import statistics
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Union
import argparse
from rich.console import Console
from rich.table import Table
import random
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

@dataclass
class RequestMetrics:
    request_id: int
    concurrent_level: int
    response_time: float
    status: str
    token_count: int
    tokens_per_second: float

@dataclass
class TestResults:
    concurrent_level: int
    avg_response_time: float
    median_response_time: float
    p95_response_time: float
    min_response_time: float
    max_response_time: float
    successful_requests: int
    failed_requests: int
    avg_tokens_per_second: float
    total_tokens: int

@dataclass
class LLMConfig:
    provider: str
    api_key: str = None
    model: str = None
    base_url: str = None

class LLMPerformanceTester:
    def __init__(self, config: LLMConfig, prompts_file: str):
        self.config = config
        self.prompts = self.load_prompts(prompts_file)
        self.console = Console()
        self.metrics: List[RequestMetrics] = []
        
        if self.config.provider == "openai":
            self.client = AsyncOpenAI(api_key=config.api_key)
        
    def load_prompts(self, prompts_file: str) -> List[str]:
        try:
            with open(prompts_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            self.console.print(f"[red]Error: {prompts_file} not found[/red]")
            return ["Tell me a short story about a robot."]  # fallback prompt
        
    async def make_request(self, session: aiohttp.ClientSession, request_id: int, concurrent_level: int) -> RequestMetrics:
        prompt = random.choice(self.prompts)
        
        if self.config.provider == "local":
            return await self._make_local_request(session, prompt, request_id, concurrent_level)
        else:
            return await self._make_openai_request(prompt, request_id, concurrent_level)
            
    async def _make_local_request(self, session: aiohttp.ClientSession, prompt: str, request_id: int, concurrent_level: int) -> RequestMetrics:
        payload = {
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False
        }
        
        start_time = time.time()
        try:
            async with session.post(f"{self.config.base_url}/v1/chat/completions", json=payload) as response:
                response_data = await response.json()
                end_time = time.time()
                
                content = response_data.get('choices', [{}])[0].get('message', {}).get('content', '')
                token_count = len(content.split())
                response_time = end_time - start_time
                tokens_per_second = token_count / response_time if response_time > 0 else 0
                
                return RequestMetrics(
                    request_id=request_id,
                    concurrent_level=concurrent_level,
                    response_time=response_time,
                    status='success',
                    token_count=token_count,
                    tokens_per_second=tokens_per_second
                )
        except Exception as e:
            print(f"Request failed: {str(e)}")
            end_time = time.time()
            return RequestMetrics(
                request_id=request_id,
                concurrent_level=concurrent_level,
                response_time=end_time - start_time,
                status='failed',
                token_count=0,
                tokens_per_second=0
            )
            
    async def _make_openai_request(self, prompt: str, request_id: int, concurrent_level: int) -> RequestMetrics:
        start_time = time.time()
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )
            end_time = time.time()
            
            content = response.choices[0].message.content
            token_count = len(content.split())  # Note: This is an approximation
            response_time = end_time - start_time
            tokens_per_second = token_count / response_time if response_time > 0 else 0
            
            return RequestMetrics(
                request_id=request_id,
                concurrent_level=concurrent_level,
                response_time=response_time,
                status='success',
                token_count=token_count,
                tokens_per_second=tokens_per_second
            )
        except Exception as e:
            print(f"OpenAI request failed: {str(e)}")
            end_time = time.time()
            return RequestMetrics(
                request_id=request_id,
                concurrent_level=concurrent_level,
                response_time=end_time - start_time,
                status='failed',
                token_count=0,
                tokens_per_second=0
            )

    async def run_concurrent_test(self, concurrent_level: int, num_requests: int) -> None:
        if self.config.provider == "local":
            async with aiohttp.ClientSession() as session:
                tasks = [
                    self.make_request(session, i, concurrent_level)
                    for i in range(concurrent_level)
                ]
                results = await asyncio.gather(*tasks)
                self.metrics.extend(results)
        else:
            tasks = [
                self.make_request(None, i, concurrent_level)
                for i in range(concurrent_level)
            ]
            results = await asyncio.gather(*tasks)
            self.metrics.extend(results)

    def calculate_results(self, concurrent_level: int) -> TestResults:
        # Get only the most recent batch of metrics for this concurrency level
        # This ensures we don't mix metrics from multiple test runs
        total_metrics = len([m for m in self.metrics if m.concurrent_level == concurrent_level])
        level_metrics = self.metrics[-total_metrics:]  # Take the last batch
        
        response_times = [m.response_time for m in level_metrics]
        successful = [m for m in level_metrics if m.status == 'success']
        
        if not response_times:
            raise ValueError(f"No metrics found for concurrency level {concurrent_level}")
        
        return TestResults(
            concurrent_level=concurrent_level,
            avg_response_time=statistics.mean(response_times),
            median_response_time=statistics.median(response_times),
            p95_response_time=statistics.quantiles(response_times, n=20)[18],
            min_response_time=min(response_times),
            max_response_time=max(response_times),
            successful_requests=len(successful),
            failed_requests=len(level_metrics) - len(successful),
            avg_tokens_per_second=statistics.mean([m.tokens_per_second for m in successful]) if successful else 0,
            total_tokens=sum(m.token_count for m in successful)
        )

    def display_results(self, results: List[TestResults], total_time: float, total_requests: int, 
                       avg_response_time: float, successful_requests: int, failed_requests: int) -> None:
        # First table - Test Results
        results_table = Table(title="LLM Performance Test Results")
        
        results_table.add_column("Concurrent\nRequests", justify="right")
        results_table.add_column("Test Time (s)", justify="right")
        results_table.add_column("Avg Time\nper Request (s)", justify="right")
        results_table.add_column("Success/Total", justify="right")
        
        for result in results:
            total_reqs = result.successful_requests + result.failed_requests
            avg_time_per_req = result.avg_response_time / result.concurrent_level
            test_time = result.max_response_time
            
            results_table.add_row(
                str(result.concurrent_level),
                f"{test_time:.2f}",
                f"{avg_time_per_req:.2f}",
                f"{result.successful_requests}/{total_reqs}"
            )
        
        # Second table - Overall Statistics
        stats_table = Table(title="\nOverall Statistics")
        
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right")
        
        stats_table.add_row("Total wall time", f"{total_time:.2f} seconds")
        stats_table.add_row("Total requests", str(total_requests))
        stats_table.add_row("Successful/Failed", f"{successful_requests}/{failed_requests}")
        stats_table.add_row("Average time per request", f"{total_time/total_requests:.2f} seconds")
        stats_table.add_row("Throughput", f"{total_requests/total_time:.2f} requests/second")
        
        # Print both tables
        self.console.print(results_table)
        self.console.print(stats_table)

async def main():
    parser = argparse.ArgumentParser(description='LLM Performance Testing Tool')
    parser.add_argument('--provider', choices=['local', 'openai'], 
                       default=os.getenv('LLM_PROVIDER', 'local'),
                       help='LLM provider to test')
    parser.add_argument('--url', default=os.getenv('LOCAL_LLM_URL', 'http://localhost:1234'),
                       help='LM Studio server URL (for local provider)')
    parser.add_argument('--model', default=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                       help='Model to use (for OpenAI provider)')
    parser.add_argument('--prompts-file', default='test_prompts.txt',
                       help='File containing test prompts')
    
    # Add mutually exclusive group for test mode
    test_mode = parser.add_mutually_exclusive_group()
    test_mode.add_argument('--single-shot', type=int, metavar='N',
                          help='Run a single test with N concurrent connections')
    test_mode.add_argument('--max-concurrent', type=int, default=10,
                          help='Maximum concurrent connections to test')
    
    parser.add_argument('--concurrent-step', type=int, default=1,
                       help='Step size between concurrent connection tests')
    parser.add_argument('--requests-per-level', type=int, default=1,
                       help='Number of times to repeat each concurrency level test')
    args = parser.parse_args()

    config = LLMConfig(
        provider=args.provider,
        api_key=os.getenv('OPENAI_API_KEY') if args.provider == 'openai' else None,
        model=args.model if args.provider == 'openai' else None,
        base_url=args.url if args.provider == 'local' else None
    )

    if config.provider == 'openai' and not config.api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return

    total_start_time = time.time()
    total_requests = 0
    
    tester = LLMPerformanceTester(config, args.prompts_file)
    
    # Handle single shot test
    if args.single_shot is not None:
        print(f"\nRunning single test with {args.single_shot} concurrent connections...")
        await tester.run_concurrent_test(args.single_shot, args.requests_per_level)
        total_requests += args.single_shot
        results = tester.calculate_results(args.single_shot)
        all_results = [results]
    else:
        # Regular incremental testing
        concurrency_levels = range(args.concurrent_step, args.max_concurrent + 1, args.concurrent_step)
        all_results = []
        
        print(f"\nTesting concurrent connections from {args.concurrent_step} to {args.max_concurrent} in steps of {args.concurrent_step}")
        
        for level in concurrency_levels:
            print(f"\nTesting with {level} concurrent connections...")
            for _ in range(args.requests_per_level):
                await tester.run_concurrent_test(level, args.requests_per_level)
                total_requests += level
            
            results = tester.calculate_results(level)
            all_results.append(results)
    
    total_time = time.time() - total_start_time
    
    # Calculate overall statistics
    all_response_times = [metric.response_time for metric in tester.metrics]
    avg_response_time = statistics.mean(all_response_times)
    successful_requests = len([m for m in tester.metrics if m.status == 'success'])
    failed_requests = len(tester.metrics) - successful_requests
    
    # Display results with both tables
    tester.display_results(
        all_results,
        total_time,
        total_requests,
        avg_response_time,
        successful_requests,
        failed_requests
    )
    
    # Save results to file
    with open('llm_performance_results.json', 'w') as f:
        # Calculate per-test metrics for JSON output
        test_details = []
        for result in all_results:
            total_reqs = result.successful_requests + result.failed_requests
            avg_time_per_req = result.avg_response_time / result.concurrent_level
            test_details.append({
                "concurrent_requests": result.concurrent_level,
                "test_time": result.max_response_time,
                "avg_time_per_request": avg_time_per_req,
                "successful_requests": result.successful_requests,
                "total_requests": total_reqs,
                "success_rate": f"{result.successful_requests}/{total_reqs}"
            })

        json.dump({
            "test_details": test_details,
            "overall_stats": {
                "total_wall_time": total_time,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": f"{successful_requests}/{total_requests}",
                "avg_time_per_request": total_time/total_requests,
                "throughput": total_requests/total_time
            },
            "raw_test_results": [asdict(r) for r in all_results]  # Keep the original detailed data
        }, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main()) 
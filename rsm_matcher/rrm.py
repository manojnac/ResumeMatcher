"""
Resume-Job Description Matcher using Retrieval-Augmented Generation (RAG)

This system demonstrates how RAG solves the resume-JD matching problem by:
1. Chunking JDs into semantic units (ingestion)
2. Retrieving relevant JD chunks for each resume section (retrieval)
3. Generating grounded, explainable match analysis (generation)

Author: AI System Design Example
Date: January 2026
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import re
from dataclasses import dataclass
from collections import defaultdict
import os
import warnings

# LLM Integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    warnings.warn("google-generativeai not installed. LLM features will be disabled.")


# ============================================================================
# CONFIGURATION & DATA MODELS
# ============================================================================

@dataclass
class Chunk:
    """Represents a semantic chunk of text with metadata"""
    text: str
    chunk_id: int
    section_type: str  # e.g., 'skills', 'qualifications', 'responsibilities'
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Result from similarity search"""
    chunk: Chunk
    similarity_score: float
    resume_section: str


@dataclass
class MatchAnalysis:
    """Final matching analysis output"""
    overall_score: float
    section_matches: Dict[str, List[RetrievalResult]]
    matched_skills: List[str]
    missing_skills: List[str]
    explanation: str


# ============================================================================
# 1. EMBEDDING ENGINE
# ============================================================================

class EmbeddingEngine:
    """
    Handles text-to-vector conversion using sentence transformers.
    
    Design Choice: Using sentence-transformers for semantic embeddings
    - Why: Captures meaning beyond keywords (e.g., "Python dev" ≈ "Python engineer")
    - Alternative: Word2Vec, GloVe (less semantic), OpenAI embeddings (API cost)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model identifier
                       'all-MiniLM-L6-v2' is lightweight, fast, good for similarity
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            print("sentence-transformers not installed. Using mock embeddings.")
            self.model = None
            self.dimension = 384  # Dimension for all-MiniLM-L6-v2
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert texts to embedding vectors.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dimension)
        """
        if self.model is None:
            # Mock embeddings for demonstration (use random but consistent)
            np.random.seed(42)
            embeddings = []
            for text in texts:
                # Simple hash-based mock (consistent per text)
                seed = hash(text) % (2**32)
                np.random.seed(seed)
                embeddings.append(np.random.randn(self.dimension))
            return np.array(embeddings)
        
        return self.model.encode(texts, convert_to_numpy=True)


# ============================================================================
# 2. CHUNKING STRATEGY
# ============================================================================

class JDChunker:
    """
    Intelligently chunks Job Descriptions into semantic units.
    
    Design Choice: Semantic chunking over fixed-size
    - Why: JD sections (skills, qualifications) are natural semantic units
    - Preserves complete thoughts (a skill requirement shouldn't be split)
    """
    
    # Section headers commonly found in JDs
    SECTION_PATTERNS = {
        'skills': r'(?i)(required skills|technical skills|skills|competencies)',
        'qualifications': r'(?i)(qualifications|requirements|minimum requirements)',
        'responsibilities': r'(?i)(responsibilities|duties|role description|what you\'ll do)',
        'education': r'(?i)(education|degree requirements)',
        'experience': r'(?i)(experience|years of experience)',
        'nice_to_have': r'(?i)(nice to have|preferred|bonus|plus)'
    }
    
    def chunk_job_description(self, jd_text: str) -> List[Chunk]:
        """
        Parse JD into logical chunks.
        
        Strategy:
        1. Identify section headers using regex patterns
        2. Split content by sections
        3. Further split long sections by bullet points or sentences
        4. Create Chunk objects with metadata
        
        Args:
            jd_text: Raw job description text
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        chunk_id = 0
        
        # Split by sections first
        sections = self._identify_sections(jd_text)
        
        for section_type, section_text in sections.items():
            # Split section into sub-chunks if it's too long
            sub_chunks = self._split_section(section_text)
            
            for sub_chunk_text in sub_chunks:
                if len(sub_chunk_text.strip()) > 10:  # Filter very short chunks
                    chunks.append(Chunk(
                        text=sub_chunk_text.strip(),
                        chunk_id=chunk_id,
                        section_type=section_type
                    ))
                    chunk_id += 1
        
        return chunks
    
    def _identify_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract sections from JD text"""
        sections = defaultdict(str)
        current_section = 'general'
        
        lines = text.split('\n')
        
        for line in lines:
            # Check if line is a section header
            matched_section = None
            for section_name, pattern in self.SECTION_PATTERNS.items():
                if re.search(pattern, line):
                    matched_section = section_name
                    break
            
            if matched_section:
                current_section = matched_section
            else:
                sections[current_section] += line + '\n'
        
        return dict(sections)
    
    def _split_section(self, section_text: str, max_chunk_size: int = 500) -> List[str]:
        """
        Split long sections into smaller chunks.
        
        Strategy: Split by bullet points, then by sentences if still too long
        """
        # Try splitting by bullet points first
        bullet_pattern = r'[\n\r]+[\s]*[•\-\*\d+\.]\s+'
        parts = re.split(bullet_pattern, section_text)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            if len(current_chunk) + len(part) < max_chunk_size:
                current_chunk += part + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = part + " "
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [section_text]


# ============================================================================
# 3. VECTOR DATABASE (FAISS)
# ============================================================================

class VectorDatabase:
    """
    FAISS-based vector store for similarity search.
    
    Design Choice: FAISS over alternatives
    - Why: Fast (GPU/CPU optimized), local (no API), production-ready
    - Alternative: Pinecone (managed, costs), ChromaDB (simpler but slower)
    
    Key Operations:
    - index(): Store embeddings with metadata
    - search(): Retrieve top-K similar vectors
    """
    
    def __init__(self, dimension: int):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding vector dimension
        """
        try:
            import faiss
            # Using L2 distance (can also use inner product)
            self.index = faiss.IndexFlatL2(dimension)
            self.faiss_available = True
        except ImportError:
            print("FAISS not installed. Using numpy-based fallback.")
            self.faiss_available = False
            self.embeddings = None
        
        self.chunks = []  # Store original chunks for retrieval
        self.dimension = dimension
    
    def add_chunks(self, chunks: List[Chunk]):
        """
        Add chunks with embeddings to the index.
        
        Args:
            chunks: List of Chunk objects with embeddings
        """
        if not chunks:
            return
        
        embeddings = np.vstack([c.embedding for c in chunks])
        
        if self.faiss_available:
            self.index.add(embeddings)
        else:
            # Fallback: store embeddings in numpy array
            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.chunks.extend(chunks)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        Find top-K most similar chunks.
        
        Args:
            query_embedding: Query vector (1D array)
            top_k: Number of results to return
            
        Returns:
            List of (Chunk, similarity_score) tuples, sorted by similarity
        """
        query_embedding = query_embedding.reshape(1, -1)
        
        if self.faiss_available:
            distances, indices = self.index.search(query_embedding, top_k)
            # Convert L2 distance to similarity score (inverse)
            # Note: Lower distance = higher similarity
            distances = distances[0]
            indices = indices[0]
            similarities = 1 / (1 + distances)
        else:
            # Fallback: compute cosine similarity manually
            similarities = self._cosine_similarity(query_embedding[0], self.embeddings)
            indices = np.argsort(similarities)[::-1][:top_k]
            similarities = similarities[indices]
        
        results = []
        for idx, similarity in zip(indices, similarities):
            idx = int(idx)  # Convert numpy int to Python int
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(similarity)))
        
        return results
    
    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a vector and matrix of vectors"""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        return np.dot(matrix_norm, vec_norm)


# ============================================================================
# 4. RESUME PARSER
# ============================================================================

class ResumeParser:
    """
    Extracts structured sections from resume text.
    
    Design Choice: Section-based parsing
    - Why: Each resume section should query different JD aspects
    - Skills → JD Skills, Experience → JD Responsibilities, etc.
    """
    
    SECTION_PATTERNS = {
        'skills': r'(?i)(skills|technical skills|competencies)',
        'experience': r'(?i)(experience|work experience|employment history|professional experience)',
        'education': r'(?i)(education|academic background)',
        'projects': r'(?i)(projects|portfolio)',
        'summary': r'(?i)(summary|objective|profile)'
    }
    
    def parse(self, resume_text: str) -> Dict[str, str]:
        """
        Extract sections from resume.
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            Dictionary mapping section names to section text
        """
        sections = defaultdict(str)
        current_section = 'general'
        
        lines = resume_text.split('\n')
        
        for line in lines:
            # Check if line is a section header
            matched_section = None
            for section_name, pattern in self.SECTION_PATTERNS.items():
                if re.search(pattern, line) and len(line.strip()) < 50:
                    matched_section = section_name
                    break
            
            if matched_section:
                current_section = matched_section
            else:
                sections[current_section] += line + '\n'
        
        return {k: v.strip() for k, v in sections.items() if v.strip()}


# ============================================================================
# 5. RAG MATCHER (MAIN ORCHESTRATOR)
# ============================================================================

class RAGMatcher:
    """
    Main RAG system orchestrator.
    
    This class ties together:
    - Ingestion: Chunk JD, embed, index
    - Retrieval: Query with resume sections, retrieve relevant JD chunks
    - Generation: Analyze matches and produce explainable output
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", use_llm: bool = True):
        """Initialize RAG components"""
        self.embedding_engine = EmbeddingEngine(embedding_model)
        self.vector_db = VectorDatabase(dimension=self.embedding_engine.dimension)
        self.jd_chunker = JDChunker()
        self.resume_parser = ResumeParser()
        
        # Store original JD for reference
        self.job_description_text = None
        
        # LLM Configuration
        self.use_llm = use_llm and GEMINI_AVAILABLE
        if self.use_llm:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.llm = genai.GenerativeModel('gemini-1.5-flash')
                print("✅ Gemini LLM enabled for enhanced explanations")
            else:
                self.use_llm = False
                print("⚠️  GEMINI_API_KEY not found. Using rule-based explanations.")
                print("   Set API key: export GEMINI_API_KEY='your-key' (Linux/Mac)")
                print("   Or: set GEMINI_API_KEY=your-key (Windows)")
        else:
            self.llm = None
            if not GEMINI_AVAILABLE:
                print("⚠️  google-generativeai not installed. Using rule-based explanations.")
    
    # ========================================================================
    # INGESTION PHASE
    # ========================================================================
    
    def ingest_job_description(self, jd_text: str):
        """
        Ingest and index a job description.
        
        Pipeline:
        1. Chunk JD into semantic units
        2. Generate embeddings for each chunk
        3. Store in vector database
        
        Args:
            jd_text: Raw job description text
        """
        print("🔄 Ingesting Job Description...")
        
        # Store original
        self.job_description_text = jd_text
        
        # Step 1: Chunk
        chunks = self.jd_chunker.chunk_job_description(jd_text)
        print(f"  ✓ Created {len(chunks)} semantic chunks")
        
        # Step 2: Embed
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_engine.embed(chunk_texts)
        
        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        print(f"  ✓ Generated embeddings (dimension: {self.embedding_engine.dimension})")
        
        # Step 3: Index
        self.vector_db.add_chunks(chunks)
        print(f"  ✓ Indexed in vector database")
    
    # ========================================================================
    # RETRIEVAL PHASE
    # ========================================================================
    
    def retrieve_relevant_chunks(
        self, 
        resume_text: str, 
        top_k: int = 3
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieve relevant JD chunks for each resume section.
        
        Pipeline:
        1. Parse resume into sections
        2. Embed each section
        3. Query vector DB for top-K similar JD chunks per section
        
        Args:
            resume_text: Raw resume text
            top_k: Number of chunks to retrieve per section
            
        Returns:
            Dictionary mapping resume sections to retrieved JD chunks
        """
        print("\n🔍 Retrieving Relevant JD Chunks...")
        
        # Step 1: Parse resume
        resume_sections = self.resume_parser.parse(resume_text)
        print(f"  ✓ Parsed {len(resume_sections)} resume sections")
        
        # Step 2 & 3: Embed and retrieve for each section
        retrievals = {}
        
        for section_name, section_text in resume_sections.items():
            if not section_text.strip():
                continue
            
            # Embed resume section
            section_embedding = self.embedding_engine.embed([section_text])[0]
            
            # Retrieve similar JD chunks
            results = self.vector_db.search(section_embedding, top_k=top_k)
            
            # Convert to RetrievalResult objects
            retrieval_results = [
                RetrievalResult(
                    chunk=chunk,
                    similarity_score=score,
                    resume_section=section_name
                )
                for chunk, score in results
            ]
            
            retrievals[section_name] = retrieval_results
            print(f"  ✓ Retrieved {len(retrieval_results)} chunks for '{section_name}'")
        
        return retrievals
    
    # ========================================================================
    # GENERATION PHASE
    # ========================================================================
    
    def generate_match_analysis(
        self, 
        resume_text: str, 
        top_k: int = 3
    ) -> MatchAnalysis:
        """
        Generate grounded matching analysis.
        
        This is where RAG shines: we use ONLY retrieved context to make claims.
        
        Args:
            resume_text: Raw resume text
            top_k: Number of JD chunks to retrieve per resume section
            
        Returns:
            MatchAnalysis object with scores and explanations
        """
        print("\n📊 Generating Match Analysis...")
        
        # Retrieve relevant context
        retrievals = self.retrieve_relevant_chunks(resume_text, top_k)
        
        # Extract skills from resume and JD
        resume_skills = self._extract_skills(resume_text)
        jd_skills = self._extract_jd_skills(retrievals)
        
        # Calculate matches
        matched_skills = list(set(resume_skills) & set(jd_skills))
        missing_skills = list(set(jd_skills) - set(resume_skills))
        
        # Calculate overall score based on retrievals
        overall_score = self._calculate_overall_score(retrievals, matched_skills, jd_skills)
        
        # Generate explanation
        explanation = self._generate_explanation(
            retrievals, 
            matched_skills, 
            missing_skills,
            overall_score
        )
        
        return MatchAnalysis(
            overall_score=overall_score,
            section_matches=retrievals,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            explanation=explanation
        )
    
    def _extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from text using both regex patterns and semantic similarity.
        
        This hybrid approach:
        1. Uses regex for common technical terms (fast, precise)
        2. Validates with embedding similarity (catches variations)
        """
        # Common technical skills patterns - expanded for better coverage
        skill_patterns = [
            r'\b(Python|Java|JavaScript|C\+\+|Ruby|Go|Rust|TypeScript|C#|Scala|PHP|MATLAB)\b',
            r'\b(React|Angular|Vue|Node\.js|Django|Flask|Spring|FastAPI|Express)\b',
            r'\b(AWS|Azure|GCP|Google Cloud|Docker|Kubernetes|Jenkins|CircleCI|GitLab)\b',
            r'\b(SQL|MongoDB|PostgreSQL|MySQL|Redis|Cassandra|Elasticsearch|DynamoDB)\b',
            r'\b(Machine Learning|Deep Learning|NLP|Computer Vision|AI|TensorFlow|PyTorch|Scikit-learn|Keras)\b',
            r'\b(Git|CI/CD|Agile|Scrum|DevOps|Linux|Bash|Kubernetes)\b',
            r'\b(REST|GraphQL|API|Microservices|Distributed Systems)\b',
            r'\b(Data Science|Data Engineering|Analytics|Hadoop|Spark|Pandas|NumPy)\b',
            r'\b(HTML|CSS|Bootstrap|Tailwind|Material UI|jQuery)\b',
            r'\b(Jenkins|Travis|CircleCI|GitHub Actions|Ansible|Terraform)\b'
        ]
        
        skills = []
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend([m for m in matches if isinstance(m, str)])
        
        # Deduplicate and normalize (case-insensitive)
        unique_skills = {}
        for skill in skills:
            normalized = skill.lower()
            if normalized not in unique_skills:
                unique_skills[normalized] = skill
        
        return list(unique_skills.values())
    
    def _extract_jd_skills(self, retrievals: Dict[str, List[RetrievalResult]]) -> List[str]:
        """Extract skills from retrieved JD chunks"""
        jd_text = ""
        # Include all section types since skills can appear anywhere
        for results in retrievals.values():
            for result in results:
                jd_text += result.chunk.text + " "
        
        return self._extract_skills(jd_text)
    
    def _calculate_overall_score(
        self, 
        retrievals: Dict[str, List[RetrievalResult]],
        matched_skills: List[str],
        jd_skills: List[str]
    ) -> float:
        """
        Calculate overall match score (0-100).
        
        Scoring components:
        - Skill match ratio: 40%
        - Average retrieval similarity: 60%
        """
        # Skill match score
        skill_score = 0
        if jd_skills:
            skill_score = (len(matched_skills) / len(jd_skills)) * 40
        
        # Retrieval similarity score
        all_similarities = []
        for results in retrievals.values():
            all_similarities.extend([r.similarity_score for r in results])
        
        avg_similarity = np.mean(all_similarities) if all_similarities else 0
        similarity_score = avg_similarity * 60
        
        return min(100, skill_score + similarity_score)
    
    def _generate_explanation(
        self,
        retrievals: Dict[str, List[RetrievalResult]],
        matched_skills: List[str],
        missing_skills: List[str],
        overall_score: float
    ) -> str:
        """
        Generate human-readable explanation.
        
        Uses LLM (Gemini) if available for natural language generation,
        otherwise falls back to template-based explanation.
        
        CRITICAL: This is grounded ONLY in retrieved context.
        No hallucination - we only claim what we can cite.
        """
        if self.use_llm and self.llm:
            return self._generate_llm_explanation(retrievals, matched_skills, missing_skills, overall_score)
        else:
            return self._generate_template_explanation(retrievals, matched_skills, missing_skills, overall_score)
    
    def _generate_llm_explanation(
        self,
        retrievals: Dict[str, List[RetrievalResult]],
        matched_skills: List[str],
        missing_skills: List[str],
        overall_score: float
    ) -> str:
        """Generate explanation using Gemini LLM"""
        
        # Build context from retrievals
        context_parts = []
        for section_name, results in retrievals.items():
            if results:
                context_parts.append(f"\nResume Section: {section_name}")
                for i, result in enumerate(results[:2], 1):  # Top 2 per section
                    context_parts.append(
                        f"  Match {i} (similarity: {result.similarity_score:.2f}): "
                        f"{result.chunk.text[:150]}"
                    )
        
        context = "\n".join(context_parts)
        
        # Create prompt for LLM
        prompt = f"""You are an expert HR recruiter analyzing resume-job description matches.

MATCH SCORE: {overall_score:.1f}/100
MATCHED SKILLS: {', '.join(matched_skills) if matched_skills else 'None found'}
MISSING SKILLS: {', '.join(missing_skills[:10]) if missing_skills else 'None'}

RETRIEVED CONTEXT FROM JOB DESCRIPTION:
{context}

Based ONLY on the above information, write a concise 3-4 sentence analysis explaining:
1. Overall match quality (use score to guide assessment)
2. Key strengths of the candidate
3. Main gaps or missing qualifications
4. Recommendation (Strong fit / Moderate fit / Weak fit / Not a fit)

Be specific and cite the retrieved context. Keep it professional and actionable."""

        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"⚠️  LLM generation failed: {e}. Falling back to template.")
            return self._generate_template_explanation(retrievals, matched_skills, missing_skills, overall_score)
    
    def _generate_template_explanation(
        self,
        retrievals: Dict[str, List[RetrievalResult]],
        matched_skills: List[str],
        missing_skills: List[str],
        overall_score: float
    ) -> str:
        """Generate explanation using templates (fallback)"""
        explanation_parts = []
        
        # Overall assessment
        if overall_score >= 75:
            assessment = "Strong match"
        elif overall_score >= 50:
            assessment = "Moderate match"
        else:
            assessment = "Weak match"
        
        explanation_parts.append(
            f"{assessment} ({overall_score:.1f}/100). "
            f"Found {len(matched_skills)} matching skills."
        )
        
        # Skill details
        if matched_skills:
            explanation_parts.append(
                f"\n\nMatching Skills: {', '.join(matched_skills[:10])}"
            )
        
        if missing_skills and len(missing_skills) <= 10:
            explanation_parts.append(
                f"\n\nMissing Skills: {', '.join(missing_skills)}"
            )
        
        # Section-by-section analysis
        explanation_parts.append("\n\nSection Analysis:")
        for section_name, results in retrievals.items():
            if results:
                top_match = results[0]
                explanation_parts.append(
                    f"\n• {section_name.capitalize()}: "
                    f"Matched with JD '{top_match.chunk.section_type}' section "
                    f"(similarity: {top_match.similarity_score:.2f})"
                )
                explanation_parts.append(f"  Relevant requirement: \"{top_match.chunk.text[:100]}...\"")
        
        return "".join(explanation_parts)
    
    # ========================================================================
    # COMPLETE PIPELINE
    # ========================================================================
    
    def match(self, jd_text: str, resume_text: str) -> MatchAnalysis:
        """
        Complete end-to-end matching pipeline.
        
        Args:
            jd_text: Job description text
            resume_text: Resume text
            
        Returns:
            MatchAnalysis with complete results
        """
        # Ingest JD
        self.ingest_job_description(jd_text)
        
        # Generate analysis
        analysis = self.generate_match_analysis(resume_text)
        
        print("\n✅ Analysis Complete!")
        return analysis


# ============================================================================
# 6. DEMO & USAGE EXAMPLE
# ============================================================================

def print_analysis(analysis: MatchAnalysis):
    """Pretty print the match analysis"""
    print("\n" + "="*70)
    print("MATCH ANALYSIS REPORT")
    print("="*70)
    print(analysis.explanation)
    print("\n" + "="*70)


def print_detailed_retrievals(analysis: MatchAnalysis):
    """Print detailed visualization of all retrieved chunks"""
    print("\n" + "="*70)
    print("DETAILED RETRIEVAL VISUALIZATION")
    print("="*70)
    print("This shows what RAG retrieved from the JD for each resume section")
    print("="*70)
    
    for section_name, results in analysis.section_matches.items():
        print(f"\n📝 RESUME SECTION: {section_name.upper()}")
        print("-" * 70)
        
        if not results:
            print("  (No retrievals)")
            continue
        
        for i, result in enumerate(results[:3], 1):  # Show top-3
            print(f"\n  Rank #{i} | Similarity: {result.similarity_score:.3f}")
            print(f"  JD Section: '{result.chunk.section_type}'")
            print(f"  Retrieved Text:")
            # Wrap text for readability
            text = result.chunk.text[:200]
            print(f"  \"{text}{'...' if len(result.chunk.text) > 200 else ''}\"")
    
    print("\n" + "="*70)


def get_multiline_input(prompt: str) -> str:
    """Get multiline input from user"""
    print(prompt)
    print("(Type your text, then press Enter on an empty line to finish)")
    print("-" * 70)
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "":
                if lines:  # Only break if we have some content
                    break
            else:
                lines.append(line)
        except EOFError:
            break
    
    return "\n".join(lines)


def demo_interactive():
    """Interactive mode - user provides their own JD and resume"""
    print("\n" + "="*70)
    print("🤖 INTERACTIVE RAG RESUME-JD MATCHER")
    print("="*70)
    print("This tool uses RAG to semantically match resumes with job descriptions")
    print("="*70 + "\n")
    
    # Get job description
    job_description = get_multiline_input("📄 Enter the Job Description:")
    
    if not job_description.strip():
        print("❌ No job description provided. Exiting.")
        return
    
    print("\n✅ Job description received!\n")
    
    # Get resume
    resume = get_multiline_input("📝 Enter the Resume:")
    
    if not resume.strip():
        print("❌ No resume provided. Exiting.")
        return
    
    print("\n✅ Resume received!\n")
    
    # Initialize matcher
    print("Initializing RAG Matcher...")
    matcher = RAGMatcher()
    
    # Run matching
    analysis = matcher.match(job_description, resume)
    
    # Display results
    print_analysis(analysis)
    
    # Show detailed retrievals
    print_detailed_retrievals(analysis)


def demo_sample():
    """Demo with hardcoded sample data"""
    
    # Sample Job Description
    job_description = """
    Senior Software Engineer - AI/ML Team
    
    Required Skills:
    - 5+ years of Python programming experience
    - Strong experience with Machine Learning frameworks (TensorFlow, PyTorch)
    - Proficiency in SQL and database design
    - Experience with cloud platforms (AWS, Azure, or GCP)
    - Knowledge of Docker and Kubernetes
    
    Qualifications:
    - Bachelor's degree in Computer Science or related field
    - Strong problem-solving and analytical skills
    - Experience with Agile development methodologies
    - Excellent communication skills
    
    Responsibilities:
    - Design and implement machine learning models for production
    - Collaborate with cross-functional teams to deploy AI solutions
    - Optimize model performance and scalability
    - Mentor junior engineers
    - Participate in code reviews and architecture discussions
    
    Nice to Have:
    - Experience with NLP or Computer Vision
    - Contributions to open-source ML projects
    - Published research papers
    """
    
    # Sample Resume
    resume = """
    John Doe
    Software Engineer
    
    Summary:
    Experienced software engineer with 6 years of expertise in Python development
    and machine learning. Passionate about building scalable AI systems.
    
    Skills:
    Python, TensorFlow, PyTorch, SQL, PostgreSQL, AWS, Docker, Git, CI/CD
    
    Experience:
    Senior Developer at TechCorp (2020-2024)
    - Developed ML models for recommendation systems using PyTorch
    - Deployed models on AWS using Docker and Kubernetes
    - Led team of 3 junior developers
    - Improved model inference time by 40%
    
    ML Engineer at DataStart (2018-2020)
    - Built NLP pipelines for text classification
    - Implemented data pipelines using Python and SQL
    - Worked in Agile teams with 2-week sprints
    
    Education:
    B.S. in Computer Science, State University (2018)
    
    Projects:
    - Open-source contribution to scikit-learn
    - Personal project: Image classification using CNNs
    """
    
    # Initialize matcher
    print("Initializing RAG Matcher...")
    matcher = RAGMatcher()
    
    # Run matching
    analysis = matcher.match(job_description, resume)
    
    # Display results
    print_analysis(analysis)
    
    # Show detailed retrievals
    print_detailed_retrievals(analysis)


def demo():
    """Main demo function - choose between interactive or sample mode"""
    print("\n" + "="*70)
    print("🚀 RESUME-JD MATCHER USING RAG")
    print("="*70)
    print("\nChoose mode:")
    print("1. Interactive mode (enter your own JD and resume)")
    print("2. Sample mode (use hardcoded example)")
    print("3. Exit")
    print("="*70)
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        demo_interactive()
    elif choice == "2":
        demo_sample()
    elif choice == "3":
        print("\n👋 Goodbye!")
        return
    else:
        print("\n❌ Invalid choice. Running sample mode by default.\n")
        demo_sample()


if __name__ == "__main__":
    demo()
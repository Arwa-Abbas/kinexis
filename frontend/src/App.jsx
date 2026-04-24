import React, { useState } from 'react'
import axios from 'axios'

const API_URL = 'http://localhost:8000/api'

function App() {
  const [videoFile, setVideoFile] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [feedback, setFeedback] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    setVideoFile(e.target.files[0])
    setFeedback(null)
    setError(null)
  }

  const analyzeVideo = async () => {
    if (!videoFile) {
      setError("Please select a video file first")
      return
    }

    setAnalyzing(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', videoFile)

    try {
      const response = await axios.post(`${API_URL}/analyze-video`, formData)
      setFeedback(response.data)
    } catch (err) {
      setError(err.response?.data?.error || "Analysis failed. Make sure backend is running.")
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: '#f4f6fb',
      fontFamily: 'Inter, sans-serif',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px'
    }}>
      <div style={{
        width: '100%',
        maxWidth: '720px',
        background: '#fff',
        borderRadius: '16px',
        padding: '28px',
        boxShadow: '0 10px 30px rgba(0,0,0,0.08)'
      }}>

        <h1 style={{ marginBottom: '6px', color: '#222' }}>
          Biomechanical Feedback
        </h1>
        <p style={{ color: '#777', marginBottom: '24px' }}>
          Upload a video to analyze movement quality
        </p>

        {/* Upload Box */}
        <div style={{
          border: '1.5px dashed #ccc',
          borderRadius: '12px',
          padding: '24px',
          textAlign: 'center',
          background: '#fafafa'
        }}>
          <input type="file" accept="video/*" onChange={handleFileChange} />

          {videoFile && (
            <p style={{ marginTop: '10px', color: '#4caf50', fontSize: '14px' }}>
              {videoFile.name}
            </p>
          )}

          <button
            onClick={analyzeVideo}
            disabled={!videoFile || analyzing}
            style={{
              marginTop: '16px',
              padding: '10px 24px',
              borderRadius: '8px',
              border: 'none',
              background: '#5b6cff',
              color: '#fff',
              fontSize: '14px',
              cursor: analyzing ? 'not-allowed' : 'pointer',
              opacity: analyzing ? 0.6 : 1,
              transition: '0.2s'
            }}
          >
            {analyzing ? 'Analyzing...' : 'Analyze'}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div style={{
            marginTop: '20px',
            padding: '12px',
            borderRadius: '8px',
            background: '#ffe6e6',
            color: '#d32f2f',
            fontSize: '14px'
          }}>
            {error}
          </div>
        )}

        {/* Results */}
        {feedback && (
          <div style={{ marginTop: '28px' }}>

            <h3 style={{ marginBottom: '16px' }}>Results</h3>

            {/* Key Metrics */}
            <div style={{
              display: 'flex',
              gap: '16px',
              justifyContent: 'space-between'
            }}>
              <div style={card}>
                <p style={label}>Left Knee</p>
                <p style={value}>{feedback.left_knee || '--'}°</p>
              </div>

              <div style={card}>
                <p style={label}>Right Knee</p>
                <p style={value}>{feedback.right_knee || '--'}°</p>
              </div>
            </div>

            {/* Feedback */}
            <div style={{
              marginTop: '16px',
              padding: '14px',
              borderRadius: '10px',
              background: feedback.feedback === 'Good form! Keep going'
                ? '#e8f5e9'
                : '#fff8e1',
              color: '#333',
              textAlign: 'center',
              fontWeight: '500'
            }}>
              {feedback.feedback}
            </div>

            {/* Table */}
            <div style={{ marginTop: '20px' }}>
              <table style={{ width: '100%', fontSize: '14px' }}>
                <tbody>
                  <tr>
                    <td style={td}>Min Angle</td>
                    <td style={td}>{feedback.min_knee || '--'}°</td>
                  </tr>
                  <tr>
                    <td style={td}>Max Angle</td>
                    <td style={td}>{feedback.max_knee || '--'}°</td>
                  </tr>
                  <tr>
                    <td style={td}>Reps</td>
                    <td style={td}>{feedback.reps || 'N/A'}</td>
                  </tr>
                  <tr>
                    <td style={td}>Symmetry</td>
                    <td style={td}>{feedback.symmetry || 'N/A'}%</td>
                  </tr>
                </tbody>
              </table>
            </div>

          </div>
        )}
      </div>
    </div>
  )
}

// reusable styles
const card = {
  flex: 1,
  background: '#f7f8fc',
  borderRadius: '12px',
  padding: '16px',
  textAlign: 'center'
}

const label = {
  fontSize: '13px',
  color: '#666'
}

const value = {
  fontSize: '28px',
  fontWeight: '600',
  marginTop: '4px'
}

const td = {
  padding: '8px 4px',
  borderBottom: '1px solid #eee'
}

export default App
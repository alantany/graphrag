import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<string>('');

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFile(event.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/upload', formData);
      setResult(response.data.result);
    } catch (error) {
      console.error('上传失败:', error);
    }
  };

  return (
    <div className="App">
      <h1>知识图谱生成系统</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>上传并处理</button>
      {result && (
        <div>
          <h2>处理结果:</h2>
          <pre>{result}</pre>
        </div>
      )}
    </div>
  );
}

export default App;